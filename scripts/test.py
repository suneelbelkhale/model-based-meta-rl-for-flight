import argparse
import os

from mbrl.experiments import logger
from mbrl.experiments.file_manager import FileManager
from mbrl.utils.file_utils import import_config
from mbrl.utils.python_utils import exit_on_ctrl_c
from mbrl.utils.tf_utils import enable_eager_execution, enable_static_execution, restore_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--gpu_frac', type=float, default=0.3)
args = parser.parse_args()


config_fname = os.path.abspath(args.config)
model_fname = os.path.abspath(args.model)
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

env_spec = params.env_spec.cls(params.env_spec.params)
env = params.env.cls(params.env.params, env_spec)
model = params.model.cls(params.model.params, env_spec, None)
policy = params.policy.cls(params.policy.params, env_spec)


### warm start the planner
obs, goal = env.reset()
policy.warm_start(model, obs, goal)


### restore model
model.restore_from_file(model_fname)


### eval loop
exit_on_ctrl_c()
done = True
while True:
    if done:
        obs, goal = env.reset()

    get_action = policy.get_action(model, obs, goal)
    obs, goal, done = env.step(get_action)
