import argparse
import os

import numpy as np

from mbrl.envs.sanity_env import CFSanityEnv
from mbrl.experiments import logger
from mbrl.experiments.file_manager import FileManager
from mbrl.utils.file_utils import import_config
from mbrl.utils.python_utils import exit_on_ctrl_c
from mbrl.utils.tf_utils import enable_eager_execution, enable_static_execution, restore_checkpoint
from mbrl.utils.torch_utils import to_torch, to_numpy

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--gpu_frac', type=float, default=0.3)
args = parser.parse_args()


config_fname = os.path.abspath(args.config)
assert os.path.exists(config_fname), '{0} does not exist'.format(config_fname)

params = import_config(config_fname)
params.freeze()

file_manager = FileManager(params.exp_name, is_continue=True)

if args.model is not None:
    model_fname = os.path.abspath(args.model)
    assert os.path.exists(model_fname), 'Model: {0} does not exist'.format(model_fname)
    logger.debug("Using model: {}".format(model_fname))
elif "checkpoint_model_file" in params.trainer.params:
    model_fname = os.path.join(file_manager.models_dir, params.trainer.params.checkpoint_model_file)
    logger.debug("Using checkpoint model for current eval: {}".format(model_fname))
else:
    model_fname = os.path.join(file_manager.models_dir, "model.pt")
    logger.debug("Using default model for current eval: {}".format(model_fname))

assert params.env.cls == CFSanityEnv

env_spec = params.env_spec.cls(params.env_spec.params)
env = params.env.cls(params.env.params, env_spec)
model = params.model.cls(params.model.params, env_spec, None)
policy = params.policy.cls(params.policy.params, env_spec)

lag = env._lag

### warm start the planner
obs, goal = env.reset()
policy.warm_start(model, obs, goal)


### restore model
model.restore_from_file(model_fname)


### eval loop
exit_on_ctrl_c()
done = True
latent_idx = -1
while True:
    if done:
        obs, goal, latent_idx = env.reset(ret_latent=True)
    obs.latent = np.ones_like(obs.latent) * latent_idx

    get_action = policy.get_action(model, obs, goal)

    print("current obs:", obs.obs[0])
    exp_o = obs.obs[0].copy()
    if lag > 0:
        if latent_idx == 1:
            exp_o[0] -= obs.prev_act[0, lag - 1, 0] * 0.1
        else:
            exp_o[0] += obs.prev_act[0, lag - 1, 0] * 0.1
        exp_o[1] += obs.prev_act[0, lag - 1, 2] * 0.1
        exp_o[2] += obs.prev_act[0, lag - 1, 1] * 0.1
    else:
        a = get_action.act[0]
        if latent_idx == 1:
            exp_o -= 2 * a[0]
        exp_o += np.array([a[0], a[2], a[1]])
    print("expected obs:", exp_o)


    # call rollout directly
    obs = obs.leaf_apply(lambda arr: to_torch(arr, device=model.device))
    goal = goal.leaf_apply(lambda arr: to_torch(arr, device=model.device))
    best = get_action.results.order[0, 0]
    results2 = policy.eval_act_sequence(model, get_action.action_sequence.leaf_apply(lambda arr: arr[:, best]), obs.copy(), goal.copy())

    # call model directly
    model.eval()
    obs.act = get_action.act
    model_out = model(obs.copy())
    model_out.leaf_modify(lambda arr: to_numpy(arr))

    obs, goal, done = env.step(get_action)

    pred_traj = get_action.results.trajectory.obs[0, best, 1]

    print("actual next obs:", obs.obs[0], "// best_pred next obs", to_numpy(pred_traj))
    print("model next obs:", model_out.next_obs[0])
    print("eval_traj next obs:", to_numpy(results2.trajectory.obs[0, 1]))

    print("")
