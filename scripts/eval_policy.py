import argparse
import math
import os

import numpy as np
import torch
from scipy.io import savemat

from mbrl.experiments import logger
from mbrl.experiments.file_manager import FileManager
from mbrl.utils.file_utils import import_config
from mbrl.utils.python_utils import exit_on_ctrl_c, AttrDict
from mbrl.utils.torch_utils import to_numpy

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str)
parser.add_argument('--output_file_base', type=str, default="policy_outputs")
parser.add_argument('--model', type=str)
parser.add_argument('--random_goals', action="store_true")
parser.add_argument('--do_train', action="store_true")
parser.add_argument('--do_holdout', action="store_true")
args = parser.parse_args()

assert args.do_train or args.do_holdout

config_fname = os.path.abspath(args.config)
assert os.path.exists(config_fname), 'Config: {0} does not exist'.format(config_fname)
params = import_config(config_fname)
params.freeze()
file_manager = FileManager(params.exp_name, is_continue=True)

if args.model is not None:
    model_fname = os.path.abspath(args.model)
    assert os.path.exists(model_fname), 'Model: {0} does not exist'.format(model_fname)
    logger.debug("Using model: {}".format(model_fname))
elif "checkpoint_model_file" in params.trainer.params:
    model_fname = os.path.join(file_manager.models_dir, params.trainer.params.checkpoint_model_file)
    logger.debug("Using checkpoint model for current experiment: {}".format(model_fname))
else:
    model_fname = os.path.join(file_manager.models_dir, "model.pt")
    logger.debug("Using default model for current experiment: {}".format(model_fname))

env_spec = params.env_spec.cls(params.env_spec.params)
# env = params.env.cls(params.env.params, env_spec)
dataset_train = params.dataset_train.cls(params.dataset_train.params, env_spec, file_manager)
dataset_holdout = params.dataset_holdout.cls(params.dataset_holdout.params, env_spec, file_manager)
model = params.model.cls(params.model.params, env_spec, None)
policy = params.policy.cls(params.policy.params, env_spec)

# ### warm start the planner TODO put back warm start
# obs, goal = env.reset()
# policy.warm_start(model, obs, goal)

### restore model
model.restore_from_file(model_fname)


### eval loop

def eval_policy(dataset, save_file_name):
    b_size = dataset.batch_size
    d_size = len(dataset)

    obs_all = []
    goals_all = []
    output_actions = []
    iters = math.ceil(d_size / b_size)
    for b in range(iters):
        logger.debug("[%d/%d]: Eval policy" % (b, iters))
        idxs = np.arange(start=b * b_size, stop=min((b + 1) * b_size, d_size))
        if args.random_goals:
            inputs, outputs = dataset.get_batch(indices=idxs, torch_device=model.device, get_horizon_goals=False)
            # this is to account for broadcasting to H+1 goals
            goals = env_spec.get_uniform(env_spec.goal_names, b_size, torch_device=model.device).unsqueeze(1)
        else:
            inputs, outputs, goals = dataset.get_batch(indices=idxs, torch_device=model.device, get_horizon_goals=True)

        # get obs batch
        obs = AttrDict()
        for name in env_spec.observation_names:
            obs[name] = inputs[name]

        act = policy.get_action(model, obs, goals, batch=True)

        goals_all.append(goals.leaf_apply(lambda v: to_numpy(v)))
        obs_all.append(obs.leaf_apply(lambda v: to_numpy(v)))
        output_actions.append(act.leaf_apply(lambda v: to_numpy(v)))

    # one big dictionary
    combined_obs = AttrDict.leaf_combine_and_apply(obs_all,
                                                   lambda vs: np.concatenate(vs, axis=0))
    combined_goals = AttrDict.leaf_combine_and_apply(goals_all,
                                                     lambda vs: np.concatenate(vs, axis=0))
    combined_output_actions = AttrDict.leaf_combine_and_apply(output_actions,
                                                              lambda vs: np.concatenate(vs, axis=0))

    combined_obs.combine(combined_goals)
    combined_obs.combine(combined_output_actions)

    logger.debug("Saving Action Sequences")
    savemat(save_file_name, combined_obs)


if args.do_train:
    eval_policy(dataset_train, os.path.join(file_manager.exp_dir, args.output_file_base + "_train.mat"))

if args.do_holdout:
    eval_policy(dataset_holdout, os.path.join(file_manager.exp_dir, args.output_file_base + "_holdout.mat"))