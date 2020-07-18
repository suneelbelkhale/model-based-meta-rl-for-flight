import argparse
import math
import os

import numpy as np
import torch
from scipy.io import savemat

from mbrl.experiments import logger
from mbrl.experiments.file_manager import FileManager
from mbrl.utils.file_utils import import_config
from mbrl.utils.mpc_utils import rollout
from mbrl.utils.python_utils import exit_on_ctrl_c, AttrDict
from mbrl.utils.torch_utils import to_numpy

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str)
parser.add_argument('--output_file_base', type=str, default="model_outputs")
parser.add_argument('--model', type=str)
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

def eval_model(dataset, save_file_name):
    b_size = dataset.batch_size
    d_size = len(dataset)

    pred_trajectories = []
    action_sequences = []
    true_trajectories = []
    costs = []

    iters = math.ceil(d_size / b_size)
    for b in range(iters):
        logger.debug("[%d/%d]: Eval model" % (b, iters))
        idxs = np.arange(start=b * b_size, stop=min((b + 1) * b_size, d_size))
        inputs, outputs, goals = dataset.get_batch(indices=idxs, torch_device=model.device, get_horizon_goals=True, get_action_seq=True)

        # get obs batch
        obs = AttrDict()
        for name in env_spec.observation_names:
            obs[name] = inputs[name]

        act_seq = AttrDict()
        act_seq['act'] = inputs['act_seq']

        model.eval()
        all_obs, all_mouts = rollout(env_spec, model, obs, act_seq, policy._advance_obs_fn)

        # first unsqueezes and then concats
        all_obs = AttrDict.leaf_combine_and_apply(all_obs,
                                                  func=lambda vs: torch.cat(vs, dim=1),
                                                  map_func=lambda arr: arr.unsqueeze(1))
        all_mouts = AttrDict.leaf_combine_and_apply(all_mouts,
                                                  func=lambda vs: torch.cat(vs, dim=1),
                                                  map_func=lambda arr: arr.unsqueeze(1))

        cost_dict = AttrDict({'costs': policy._cost_fn(all_obs, goals, act_seq, all_mouts)})

        true_trajectories.append(goals.leaf_apply(lambda v: to_numpy(v)))
        pred_trajectories.append(all_obs.leaf_apply(lambda v: to_numpy(v)))
        action_sequences.append(act_seq.leaf_apply(lambda v: to_numpy(v)))
        costs.append(cost_dict.leaf_apply(lambda v: to_numpy(v)))

    # one big dictionary
    final_dict = AttrDict.leaf_combine_and_apply(true_trajectories,
                                                 lambda vs: np.concatenate(vs, axis=0))
    combined_pred = AttrDict.leaf_combine_and_apply(pred_trajectories,
                                                    lambda vs: np.concatenate(vs, axis=0))
    combined_acts = AttrDict.leaf_combine_and_apply(action_sequences,
                                                    lambda vs: np.concatenate(vs, axis=0))
    combined_costs = AttrDict.leaf_combine_and_apply(costs,
                                                     lambda vs: np.concatenate(vs, axis=0))

    final_dict.combine(combined_pred)
    final_dict.combine(combined_acts)  # no overlapping keys
    final_dict.combine(combined_costs)

    logger.debug("Saving Model Trajectories")
    logger.debug("Keys: " + str(final_dict.keys()))
    savemat(save_file_name, final_dict)


if args.do_train:
    eval_model(dataset_train, os.path.join(file_manager.exp_dir, args.output_file_base + "_train.mat"))

if args.do_holdout:
    eval_model(dataset_holdout, os.path.join(file_manager.exp_dir, args.output_file_base + "_holdout.mat"))