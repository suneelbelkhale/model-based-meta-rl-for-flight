# Model-Based Meta-Reinforcement Learning for Flight with Suspended Payloads

## Abstract: 

Transporting suspended payloads is challenging for autonomous aerial vehicles because the payload can cause significant and unpredictable changes to the robot's dynamics. These changes can lead to suboptimal flight performance or even catastrophic failure. Although adaptive control and learning-based methods can in principle adapt to changes in these hybrid robot-payload systems, rapid mid-flight adaptation to payloads that have a priori unknown physical properties remains an open problem. We propose a meta-learning approach that "learns how to learn" models of altered dynamics within seconds of post-connection flight data. Our experiments demonstrate that our online adaptation approach outperforms non-adaptive methods on a series of challenging suspended payload transportation tasks. Videos and other supplemental material are available on our website: https://sites.google.com/view/meta-rl-for-flight.

## System Requirements

This repository was developed and has been tested on the following system:

- Ubuntu 16.04
- CUDA 9.2
- ROS Kinetic
- Python 3.6

Python package requirements are found in ROOT/requirements.txt. After installing ROS to work with python3 and getting `catkin_pkg`, run `pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html`. Note that running the offline and online Tello environments requires ROS to be installed with our crazyflie package (latest tested commit: 964916bbf83f0b8b7a155904cc8e52848855848d). The source for this repository can be found here: https://github.com/gkahn13/crazyflie. This package should be built with catkin and sourced.

## Code Overview

This repository is structured as follows:

```
 ├──configs                                 - Experiment configuration: most changes occur here by design.
 ├──data                                    - Location for storing datasets (optional).
 ├──experiments                             - Experiment log file root: each experiment gets its own folder.
 ├──scripts                                 - Main experiment scripts: nothing else should be run directly.
 ├──src                                     - Latent PETS source.
 │    ├──mbrl                               
 │    │    ├──datasets                      - Datasets used to get batches of data
 │    │    ├──envs                          - Gym-like environments and environment specifications
 │    │    ├──experiments                   - File manager and logger
 │    │    ├──models                        - PyTorch models that are used for training
 │    │    ├──policies                      - Policies (e.g. CEM, Random Shooting) to get actions for stepping an env.
 │    │    ├──sandbox                       - <ignore>
 │    │    ├──trainers                      - Training loops over dataset(s) given a model, env(s), and a policy.
 │    │    ├──utils                         - Shared code functionalities
```

## Configs

Configuration files set up the full experiment through a modified dictionary (AttrDict) defined in `src/utils/`. The same config can be used for training and inference, based on the script that is called. Currently, there are four example configurations that have each been tested. The `latent_*` configs represents configurations with latent variable training. The configurations use either the tello_pendulum_controller or the sanity environments defined under `src/envs/`.

By convention, all the parameters for a given class are specified in the `__init__` function at the beginning of the class file. The example configs should require minimal modification to run.

All configs specify an experiment folder. In each folder after training, there will be a folder for checkpoint models (if this is specified in the trainer params), csv log of the losses, a tensorboard-compatible event log, a copy of the used config, and exact git versions (with diffs). Configs let you specify the model file you want to use during testing (`MODEL_FILE`)


## Scripts

- **eval_model.py**: Recover direct model outputs on some input dataset.
- **eval_policy.py**: Recover policy outputs on some input dataset.
- **mat_latent_legacy_converter.py**: Rescale the latent variables to be 0 -> N in a given mat file. This was necessary for some of the older tello datasets when converting to the new repository.
- **sanity_test.py**: Used with the sanity environment to debug latent inference.
- **test.py**: Evaluate a policy/config with no latents.
- **test_latent_inference.py**: Evaluate a policy/config with latent inference.
- **train.py**: Run a trainer with a given configuration. Use `--continue` to resume training.


## Datasets

All experiments currently use the MatDataset, which requires a specific formatting that can be referenced in the `split_data_by_episodes()` function in `src/utils/data_utils.py`. By specifying `None` as the input file, this dataset becomes an "online" dataset, meaning it can be used during latent inference testing to record transitions in each episode.

There are publicly available datasets for training models with this repository, and the file paths can be changed at the top of the config files:

https://drive.google.com/drive/folders/1F0HLhIWc3MVBxkgOve2tDCL81vfJ3__R?usp=sharing

For tello files, you should use the `mapped_` datasets (these have been modified with the legacy converter). We recommend just using the `tello/mapped_TELLO_DT_0_25_LATENT_dec12.mat` for training.

## ROS

For all testing and inference with the tello, we use ROS for communication. The crazyflie repository should be cloned in your `catkin_ws/src/` and then made with `catkin_make`. Make sure the correct python version is specified. Since we use cv_bridge, you will also have to clone cv_bridge and make this locally as well to override the default python2 version. The crazyflie package comes with a visualization tool, which can be run as follows:

`rosrun crazyflie pendulum_visualization_node.py _id:=0 _num_trajectories:=10 _dt:=0.25`

The latent_pets repository publishes messages (`ros_copter_env`) during inference that are visualized by this node. Additionally, the image taken from the external camera will appear here once the rosbag or live stream begins playing.

For offline inference (playing pre-recorded rosbags of the robot), you will need to download one of our example rosbags, which can be found here:

https://drive.google.com/drive/folders/1yv4hoqmA-FYwVeD82idDyCftcC0rjdhc?usp=sharing

These rosbags are some of the experiments that we ran for the paper, and each follow a goal "trajectory".

## Example

Here, we outline a flow for training with latent variables (`configs/latent_tello_config.py`).

1. First, make sure to look through the config file and see what each parameter is responsible for in the corresponding class. For convenience, the configs have a `USE_LATENT` toggle to disable the latent training / testing. To use known latents (hard coded values as specified in the dataset), set `UNKNOWN_LATENT = False`. This outline assumes we are training unknown latents (learning latent variables during model training). The basic experiment uses 2 latent classes (18CM and 30CM string length). Update the experiment folder to a new location.

2. From scripts, run `python train.py ../configs/latent_tello_config.py`. With the given config, this will train the dynamics model with unknown latents for 100000 steps.

3. After/during training, open up tensorboard in a new terminal: `tensorboard --logdir <exp_folder>`. Here, you can monitor the base loss (log probability of the PETS model), the latent loss (log probability of PETS model + KL of the latent distributions with a standard normal), and the individual values of each latent distribution as well. Ideally the means of the 2 latent classes diverge sufficiently and the standard deviation reduces. Pick a reasonable training point using the holdout loss, and specify this checkpoint model as `MODEL_FILE = chkpt_<iter>.pt` in the config (can override the value that is currently there). This specifies what file to use for running inference. If this is not specified, the default will be `model.pt` (the latest).

4. For setting up running offline inference (from a rosbag), you will need to follow the steps in the ROS section. Additionally, make sure `OFFLINE=True` in the config file (which ensures the inference dataset stores true actions from the rosbag instead of current policy actions), and the control path is set appropriately based on the rosbag (e.g. `CONTROL = 'box'`). As a test, download `TEST_18CM_box_tpc_LATENT_dec12_DE_relhist_ac_obhist8_sqrt/rosbag0000.bag`.

5. First play this rosbag in a new terminal just to make sure nothing breaks. Then rosbag play again and run the visualization node specified earlier and make sure the image of the tello appears. For this test file, we want to run the rosbag without publishing the inference ros topics (anything prefixed with `mpc/`), so run rosbag play again with the visualization node, and you should just see the image of the tello: `rosbag play -l TEST_18CM_box_tpc_LATENT_dec12_DE_relhist_ac_obhist8_sqrt/rosbag0000.bag /mpc/action_marker:=/null/0 /mpc/action_vector:=/null/1 /mpc/goal_vector:=/null/2 /mpc/latent_vector:=/null/3 /mpc/loss_vector:=/null/4 /mpc/reward_vector:=/null/5 /mpc/trajectory_marker:=/null/6`. This is a crude way to just mute the mpc topics and only play the state measurements in a loop.

6. Once this bag is playing properly, run `python test_latent_inference.py ../configs/latent_tello_config.py` to begin inference. The additional graphs in the visualization node should begin to display the inferred actions and latents.
