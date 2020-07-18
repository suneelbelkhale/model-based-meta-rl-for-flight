import os

import numpy as np
import torch

from mbrl.datasets.mat_dataset import MatDataset
from mbrl.envs.latent_env_spec import LatentEnvSpec
from mbrl.envs.sanity_env import CFSanityEnv
from mbrl.experiments.file_manager import FileManager
from mbrl.models.latent_model import LatentModel
from mbrl.policies.cem import CEM
from mbrl.policies.latent_mpc_policy import LatentMPCPolicy
from mbrl.policies.random_shooting import RandomShooting
from mbrl.trainers.latent_inference_trainer import LatentInferenceTrainer
from mbrl.trainers.latent_trainer import LatentTrainer
from mbrl.utils.mpc_utils import latent_advance_obs_fn, default_mpc_cost_fn, latent_obs_to_output_obs_fn
from mbrl.utils.param_utils import LayerParams, SequentialParams
from mbrl.utils.python_utils import AttrDict as d

# BASE EXPERIMENT PARAMS
from mbrl.utils.torch_utils import log_gaussian_prob, advance_history

obs_dim = 3
act_dim = 3
HORIZON = 5
OBS_HISTORY_LENGTH = 1
ACT_HISTORY_LENGTH = 1
DATA_INPUT_TRAIN = '~/latent_pets/data/cf_sanity/cf_sanity_lag1_2latent.mat'
DATA_INPUT_HOLDOUT = '~/latent_pets/data/cf_sanity/cf_sanity_lag1_2latent_val.mat'
DEVICE = "cuda:0"

# PETS
PROBABILISTIC = False
NUM_NETS = 1

# Latent
USE_LATENT = True
UNKNOWN_LATENT = True
NUM_LATENT_CLASSES = 2
LATENT_DIM = 1
DEFAULT_LATENT_MU = None
DEFAULT_LATENT_LOG_SIGMA = None
LATENT_TRAIN_EVERY_N = 0  # default is don't train

if USE_LATENT:
    # unknown latent trains, otherwise we don't train
    if UNKNOWN_LATENT:
        LATENT_TRAIN_EVERY_N = 1
    # known latent setup
    else:
        DEFAULT_LATENT_MU = [
            np.zeros(LATENT_DIM),
            np.ones(LATENT_DIM),
        ]
        DEFAULT_LATENT_LOG_SIGMA = - 100.0 * np.ones(LATENT_DIM)


# model stuff
MODEL_IN = obs_dim * (1 + OBS_HISTORY_LENGTH) + act_dim * (1 + ACT_HISTORY_LENGTH) + LATENT_DIM
MODEL_OUT = obs_dim * 2 if PROBABILISTIC else obs_dim

MODEL_FILE = "chkpt_0024000.pt"

# env stuff
DT = 0.1  # from env
STEP_DT = 0.5  # rendering freq
LAG = 1


###############  FUNCTIONS  ##############


def preproc_fn(inputs: d) -> d:
    assert obs_dim == 3, "preproc function out of date"
    # inputs.obs = torch.cat([inputs.obs[:, :2], torch.sqrt(inputs.obs[:, 2:])],
    #                        dim=1)  # converts to side length before model is called
    return inputs


def postproc_fn(inputs: d, model_outputs: d) -> d:
    # converts delta to absolute, converts (N, obs_dim) to (N, 1, obs_dim)
    model_outputs.next_obs += inputs.obs.float().unsqueeze(1)  # side length + delta side length
    # inputs.obs = torch.cat([inputs.obs[:, :2], inputs.obs[:, 2:] ** 2], dim=1)  # converts back to area after model gets called
    # model_outputs.next_obs = torch.cat(
    #     [model_outputs.next_obs[:, :, :2], model_outputs.next_obs[:, :, 2:] ** 2], dim=2)
    return model_outputs


def loss_fn(inputs: d, outputs: d, model_outputs: d) -> torch.Tensor:
    return -log_gaussian_prob(model_outputs.next_obs, model_outputs.next_obs_sigma, outputs.next_obs.unsqueeze(1))


# converts the obs to the output obs form
obs_to_output_obs_fn = latent_obs_to_output_obs_fn

# returns the next OBSERVATION, not to be confused with inputs
advance_obs_fn = latent_advance_obs_fn

mpc_cost_fn = default_mpc_cost_fn


############### START PARAMS ##############


def get_env_spec_params():
    obs_range = (np.zeros(3), np.array([1, 1, 0.5]))
    act_range = (np.array([-0.5, -0.5, -0.4]), np.array([0.5, 0.5, 0.4]))
    return d(
        cls=LatentEnvSpec,
        params=d(
            names_shapes_limits_dtypes=[
                ('obs', (obs_dim,), obs_range, np.float32),
                ('prev_obs', (OBS_HISTORY_LENGTH, obs_dim), obs_range, np.float32),
                ('prev_act', (ACT_HISTORY_LENGTH, act_dim), act_range, np.float32),
                ('latent', (LATENT_DIM,), (0, NUM_LATENT_CLASSES - 1), np.int),

                ('next_obs', (obs_dim,), obs_range, np.float32),
                ('next_obs_sigma', (obs_dim,), (0, np.inf), np.float32),

                ('goal_obs', (HORIZON+1, obs_dim), obs_range, np.float32),

                ('act', (act_dim,), act_range, np.float32),
            ]
        )
    )


def get_env_params():
    return d(
        cls=CFSanityEnv,
        params=d(
            dt=DT,
            step_dt=STEP_DT,
            ros_prefix="cf/0/",
            lag=LAG,
            use_random_goal=True,
            num_latent=NUM_LATENT_CLASSES,
            obs_hist_len=OBS_HISTORY_LENGTH,
            act_hist_len=ACT_HISTORY_LENGTH,
            horizon=HORIZON,
        )
    )


def get_dataset_params(input_file):
    return d(
        cls=MatDataset,
        params=d(
            input_file=input_file,
            output_file='',
            batch_size=100,
            planning_horizon=HORIZON,
            obs_history_length=OBS_HISTORY_LENGTH,
            acs_history_length=ACT_HISTORY_LENGTH,
        )
    )


def get_inference_dataset_params():
    return d(
        cls=MatDataset,
        params=d(
            input_file=None,  # empty datadict
            output_file=None,  # no save
            batch_size=16,
            planning_horizon=HORIZON,
            obs_history_length=OBS_HISTORY_LENGTH,
            acs_history_length=ACT_HISTORY_LENGTH,
        )
    )


def get_model_params():
    return d(
        cls=LatentModel,
        params=d(
            device=DEVICE,
            preproc_fn=preproc_fn,
            postproc_fn=postproc_fn,
            loss_fn=loss_fn,
            num_nets=NUM_NETS,
            is_probabilistic=PROBABILISTIC,
            deterministic_sigma_multiplier=0.01,  # default sigma_obs uncertainty multiplier
            network=SequentialParams([
                LayerParams("linear", in_features=MODEL_IN, out_features=200), LayerParams('relu'),
                LayerParams("linear", in_features=200, out_features=200), LayerParams('relu'),
                LayerParams("linear", in_features=200, out_features=200), LayerParams('relu'),
                LayerParams("linear", in_features=200, out_features=NUM_NETS * MODEL_OUT),
            ]),
            latent_object=d(
                device=DEVICE,
                num_latent_classes=NUM_LATENT_CLASSES,
                latent_dim=LATENT_DIM,
                known_latent_default_mu=DEFAULT_LATENT_MU,
                known_latent_default_log_sigma=DEFAULT_LATENT_LOG_SIGMA,
                beta_kl=0.1,  # weighting of the KL term relative to logprob in inference
            ),
        )
    )


def get_trainer_params():
    return d(
        cls=LatentTrainer,
        params=d(
            dynamics_learning_rate=1e-4,
            latent_learning_rate=5e-4,
            latent_train_every_n_steps=LATENT_TRAIN_EVERY_N,  # 0 for fixed latent
            sample_every_n_steps=0,
            train_every_n_steps=1,
            holdout_every_n_steps=500,
            max_steps=1e5,
            max_train_data_steps=0,
            max_holdout_data_steps=0,
            log_every_n_steps=1e3,
            save_every_n_steps=1e3,
            checkpoint_model_file=MODEL_FILE,
            save_checkpoints=True,
        )
    )

def get_inference_trainer_params():
    return d(
        cls=LatentInferenceTrainer,
        params=d(
            train_every_n_steps=1 if USE_LATENT else 0,
            latent_learning_rate=5e-4,
            log_every_n_steps=1e2,
            save_every_n_steps=0,
            train_min_buffer_size=2,
            max_steps_per_rollout=100,
            obs_to_output_obs_fn=obs_to_output_obs_fn,
        )
    )

def get_policy_params():
    return d(
        cls=LatentMPCPolicy,
        params=d(
            num_particles=None,
            horizon=HORIZON,
            cost_function=mpc_cost_fn,
            advance_obs_function=advance_obs_fn,
            optimizer_cls=CEM,
            optimizer_params=d(
                popsize=50,
                horizon=HORIZON,
                act_dim=act_dim,
                max_iters=3,
                num_elites=10,
                epsilon=0.001,
                alpha=0.25,
            )
        )
    )


params = d(
    exp_name='cf_sanity/lag1/KL_unknown_latent_2class',

    env_spec=get_env_spec_params(),
    env=get_env_params(),
    dataset_train=get_dataset_params(DATA_INPUT_TRAIN),
    dataset_holdout=get_dataset_params(DATA_INPUT_HOLDOUT),
    dataset_inference=get_inference_dataset_params(),
    model=get_model_params(),
    trainer=get_trainer_params(),
    inference_trainer=get_inference_trainer_params(),
    policy=get_policy_params(),
)
