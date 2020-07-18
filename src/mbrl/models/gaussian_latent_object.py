import numpy as np
import torch
from torch import nn
import torch.distributions as D

from mbrl.experiments import logger
from mbrl.models.model import Model
from mbrl.utils import abstract
from mbrl.utils.python_utils import AttrDict
from mbrl.utils.torch_utils import log_gaussian_prob, kl_regularization


class GaussianLatentObject(Model):

    @abstract.overrides
    def _init_params_to_attrs(self, params):
        self._num_latent_classes = params.num_latent_classes  # latent classes
        self._latent_dim = params.latent_dim

        self._beta_kl = params.beta_kl

        known_latent_mu = params.known_latent_default_mu
        known_latent_log_sigma = params.known_latent_default_log_sigma

        if known_latent_mu is None or known_latent_log_sigma is None:
            logger.warn("[Gaussian Latent]: Assuming unknown latent values")
            self._unknown_latent = True

            known_latent_mu = np.zeros((self._num_latent_classes, self._latent_dim))
            known_latent_log_sigma = np.zeros((self._num_latent_classes, self._latent_dim))

        self._latent_default_mu = np.broadcast_to(np.array(known_latent_mu, dtype=np.float32),
                                                  (self._num_latent_classes, self._latent_dim))
        self._latent_default_log_sigma = np.broadcast_to(np.array(known_latent_log_sigma, dtype=np.float32),
                                                         (self._num_latent_classes, self._latent_dim))

        # mean and log_sigma midpoints are used for the online mean
        self._online_latent_default_mu = self._latent_default_mu.mean(axis=0)
        self._online_latent_default_log_sigma = np.zeros(self._latent_dim, dtype=np.float32)

    @abstract.overrides
    def _init_setup(self):
        # TODO these might get overriden by model restore...
        # noinspection PyArgumentList
        self.online_mu = nn.Parameter(torch.from_numpy(self._online_latent_default_mu).to(self.device))
        # noinspection PyArgumentList
        self.online_log_sigma = nn.Parameter(torch.from_numpy(self._online_latent_default_log_sigma).to(self.device))

        for i in range(self._num_latent_classes):
            start_mu = np.ones(self._latent_dim, dtype=np.float32) * self._latent_default_mu[i]
            start_log_sig = np.ones(self._latent_dim, dtype=np.float32) * self._latent_default_log_sigma[i]
            # noinspection PyArgumentList
            self.__setattr__("mu_%d" % i, nn.Parameter(torch.from_numpy(start_mu).to(self.device)))
            # noinspection PyArgumentList
            self.__setattr__("log_sigma_%d" % i, nn.Parameter(torch.from_numpy(start_log_sig).to(self.device)))

    @abstract.overrides
    def warm_start(self, model, observation, goal):
        raise NotImplementedError

    def get_online_latent_mu_logsig(self):
        return self.online_mu, self.online_log_sigma

    def get_latent_mu_logsig(self):
        mus, logsigs = [], []
        for i in range(self._num_latent_classes):
            mus.append([getattr(self, "mu_%d" % i)])
            logsigs.append([getattr(self, "log_sigma_%d" % i)])
        return mus, logsigs

    def reset_online_latent_mu_logsig(self):
        self.online_mu.data = torch.from_numpy(self._online_latent_default_mu).to(self.device)
        self.online_log_sigma.data = torch.from_numpy(self._online_latent_default_log_sigma).to(self.device)

    def loss(self, inputs, outputs, get_model_out, i=0, writer=None, writer_prefix=""):
        distributions = self(inputs)
        inputs.latent = distributions.sample
        model_outputs = get_model_out(inputs)
        loss, logprob, kl = self._get_latent_loss(distributions['mu'], distributions['log_sigma'],
                                     model_outputs['next_obs'], model_outputs['next_obs_sigma'],
                                     outputs['next_obs'].unsqueeze(-2))

        if writer:
            writer.add_scalar(writer_prefix + "latent_loss", loss.item(), i)
            writer.add_scalar(writer_prefix + "latent_loss_kl", kl.item(), i)
            writer.add_scalar(writer_prefix + "latent_loss_logprob", logprob.item(), i)

            for j in range(self._latent_dim):
                writer.add_scalar(writer_prefix + "online_latent_mu_dim=%d" % j, self.online_mu[j].item(), i)
                writer.add_scalar(writer_prefix + "online_latent_log_sigma_dim=%d" % j, self.online_log_sigma[j].item(), i)

            mus, logsigs = self.get_latent_mu_logsig()
            for d, (mu, lsig) in enumerate(zip(mus, logsigs)):
                for j in range(self._latent_dim):
                    writer.add_scalar(writer_prefix + "latent_%d_mu_dim=%d" % (d, j), mu[j].item(), i)
                    writer.add_scalar(writer_prefix + "latent_%d_log_sigma_dim=%d" % (d, j), lsig[j].item(), i)

        return loss

    def _get_latent_loss(self, mu_lat, logs_lat, mu_next_obs, sigma_next_obs, targ_next_obs):
        log_prob = log_gaussian_prob(mu_next_obs, sigma_next_obs, targ_next_obs)  # P(s' | s, a, z)
        kl = kl_regularization(mu_lat, logs_lat)  # KL(q_phi || N(0,1))

        return - log_prob + self._beta_kl * kl, log_prob, kl

    @abstract.overrides
    def forward(self, inputs, obs_lowd=None):
        """
        Given inputs, map them to the appropriate latent distribution

        :param inputs (AttrDict): holds obs, prev_obs, prev_act, latent and "act"
        :param training (bool):
        :return: AttrDict: parametrizes distribution of latents, holds mu, log_sigma
        """

        assert hasattr(inputs, 'latent')
        assert inputs.latent.dtype in [torch.short, torch.int, torch.long], \
            "Latent is type: " + str(inputs.latent.type())
        orig = inputs.latent.view(inputs.latent.shape[0])  # should be (batch, 1)
        # map latent classes to mu, log_sig
        mus = []
        log_sigs = []
        for latent_class in orig:
            # -1 class specifies online inference
            if latent_class.item() == -1:
                mus.append(self.online_mu)
                log_sigs.append(self.online_log_sigma)
            else:
                mus.append(self.__getattr__("mu_%d" % latent_class.item()))
                log_sigs.append(self.__getattr__("log_sigma_%d" % latent_class.item()))

        mu = torch.stack(mus)
        log_sigma = torch.stack(log_sigs)

        # torch_distribution = D.normal.Normal(loc=mu, scale=log_sigma.exp())
        # sample = torch_distribution.rsample()  # sample from latent diagonal gaussian (reparam trick for gradient)
        sample = mu + torch.randn_like(mu) * log_sigma.exp()

        return AttrDict({
            'mu': mu,
            'log_sigma': log_sigma,
            'sample': sample
        })

    # @property
    # def torch_default_sigma_obs(self):
    #     return self._torch_default_sigma_obs

