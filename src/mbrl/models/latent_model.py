import numpy as np
import torch
from torch import nn

from mbrl.experiments import logger
from mbrl.models.gaussian_latent_object import GaussianLatentObject
from mbrl.models.model import Model
from mbrl.utils import abstract
from mbrl.utils.python_utils import AttrDict
from mbrl.utils.torch_utils import log_gaussian_prob, to_numpy


class LatentModel(Model):

    @abstract.overrides
    def _init_params_to_attrs(self, params):
        self._num_nets = params.num_nets
        self._is_probabilistic = params.is_probabilistic
        self._net = params.network.to_module_list(as_sequential=True).to(self.device)
        self._latent_params = params.latent_object
        # multiply with dataset sigma obs to get sigma_obs for deterministic case
        self._deterministic_sigma_multiplier = float(params.deterministic_sigma_multiplier)
        assert self._deterministic_sigma_multiplier > 0

        self._all_input_keys = self._env_spec.observation_names + self._env_spec.action_names

    @abstract.overrides
    def _init_setup(self):
        if self._dataset_train is not None:
            output_stats = self._dataset_train.get_output_stats()
        else:
            output_stats = AttrDict({'mu': np.array([0]), 'sigma': np.array([1])})
            output_stats.leaf_modify(lambda arr: np.broadcast_to(arr, self._env_spec.names_to_shapes.obs))
        logger.debug("[Latent Model] obs stats: mu = %s, sigma = %s" %
                     (output_stats.mu, output_stats.sigma))
        self.output_stats = output_stats

        # default deterministic confidence: standard deviation of (next_obs - obs) div by 10
        # noinspection PyArgumentList
        self._default_sigma_obs = nn.Parameter(torch.from_numpy(output_stats.sigma.astype(float)).to(self.device),
                                               requires_grad=False)

        self._latent_obj = GaussianLatentObject(self._latent_params, self._env_spec, self._dataset_train)
        assert hasattr(self, '_preproc_fn')
        assert hasattr(self, '_postproc_fn')
        assert hasattr(self, '_loss_fn')

    def latent_parameters(self):
        return self._latent_obj.parameters()

    def base_parameters(self):
        return self._net.parameters()

    # LATENT OBJECT STUFF
    def get_latent_mu_logsig(self):
        return self._latent_obj.get_latent_mu_logsig()

    def get_online_latent_mu_logsig(self):
        return self._latent_obj.get_online_latent_mu_logsig()

    def reset_latent_model(self):
        self._latent_obj.reset_online_latent_mu_logsig()

    def latent_loss(self, inputs, outputs, i=0, writer=None, writer_prefix=""):
        return self._latent_obj.loss(inputs, outputs, self._forward, i=i, writer=writer, writer_prefix=writer_prefix)

    def base_loss(self, inputs, outputs, i=0, writer=None, writer_prefix=""):
        model_outputs = self(inputs)
        if i > 0 and i % 1000 == 0:
            logger.debug("-------------------------------------------------")
            diff = to_numpy(model_outputs.next_obs - outputs.next_obs)
            # logger.debug("L1 loss: {}, L2^2 loss: {}".format((torch.abs(diff).sum().item() / inputs.obs.shape[0]),
            #        ((diff **2).sum().item() / inputs.obs.shape[0]))) logger.debug("Default Sigma: {}".format(
            # self._latent_obj.torch_default_sigma_obs))
            logger.debug("Mu0: {}".format(self._latent_obj.mu_0.data))
            # logger.debug("Logsig0: {}".format(self._latent_obj.log_sigma_0.data))
            logger.debug("Mu1: {}".format(self._latent_obj.mu_1.data))
            # logger.debug("Logsig1: {}".format(self._latent_obj.log_sigma_1.data))
            logger.debug(
                "SHAPES: Obs {}, Act {}, Next Obs {}, Pred Next Obs {}".format(inputs.obs.shape, inputs.act.shape,
                                                                               outputs.next_obs.shape,
                                                                               model_outputs.next_obs.shape))
            logger.debug("VALS  : Obs {}, Act {}, Next Obs {}, Pred Next Obs {}".format(to_numpy(inputs.obs[0]),
                                                                                        to_numpy(inputs.act[0]),
                                                                                        to_numpy(outputs.next_obs[0]),
                                                                                        to_numpy(
                                                                                            model_outputs.next_obs[0])))
            logger.debug("PRED ERROR: {}".format(diff[0, 0]))
            logger.debug("SCALED ERROR: {}".format(diff[0, 0] / self.output_stats.sigma_delta))
            logger.debug("-------------------------------------------------")

        loss = self._loss_fn(inputs, outputs, model_outputs)

        if writer:
            writer.add_scalar(writer_prefix + "base_loss", loss.item(), i)

        return loss

    @abstract.overrides
    def warm_start(self, model, observation, goal):
        raise NotImplementedError

    def _forward(self, inputs):
        inputs = self._preproc_fn(inputs)

        # defines the model input ordering
        arrays = []
        for key in self._env_spec.observation_names + self._env_spec.action_names:
            # print(key, inputs[key].shape)
            arrays.append(inputs[key].view(inputs[key].shape[0], -1).float().to(self.device))

        torch_in = torch.cat(arrays, dim=1)  # (batch, in_size)
        batch_size = torch_in.shape[0]

        torch_out = self._net(torch_in).view(batch_size, self._num_nets, -1)  # (b, num_models, out_size)
        if self._is_probabilistic:
            torch_mu, torch_sigma = torch.chunk(torch_out, 2, dim=2)  # 2 outputs per model (b, num_models, out_size)
        else:
            torch_mu = torch_out
            torch_sigma = torch.ones_like(torch_mu) * \
                          (self._default_sigma_obs.float() * self._deterministic_sigma_multiplier)

        outputs = AttrDict({
            "next_obs": torch_mu,
            "next_obs_sigma": torch_sigma,
        })

        return self._postproc_fn(inputs, outputs)

    @abstract.overrides
    def forward(self, inputs, obs_lowd=None):
        """
        :param latent_training:
        :param training:
        :param inputs (AttrDict): holds obs, prev_obs, prev_act, latent and "act"
        :param obs_lowd
        :param training (bool):
        :return: AttrDict
        """

        inputs = inputs.copy()
        if inputs.latent.numel() > 0:
            dist = self._latent_obj(inputs)
            inputs.latent = dist.sample

        return self._forward(inputs)
