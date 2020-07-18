import numpy as np

# gets previous n element stacked mat (B, N) -> (B, N * n_prev))
# stacked in previous order (-1, -2, -3, -4 ...)
from mbrl.experiments import logger
from mbrl.utils.np_utils import split_dim_np
from mbrl.utils.torch_utils import split_dim


def fill_n_prev(mat, n_prev, initial_zero=True):
    if initial_zero:
        prev_filled_mat = np.zeros((mat.shape[0], n_prev * mat.shape[1]))  # prev starts at initial value
    else:
        prev_filled_mat = np.tile(mat, (1, n_prev))  # prev starts at initial value
    for i in range(n_prev):
        # fill in column block with previous row (from rows i+1 down)
        if i + 1 < mat.shape[0]:
            start_col = i * mat.shape[1]
            end_col = (i + 1) * mat.shape[1]
            # block of size [B - (i+1), N]
            prev_filled_mat[i + 1:, start_col:end_col] = mat[:-(i + 1)]

    return prev_filled_mat


# splits raw mat file  input data into obs list, acs list, latent list, and various seq lists, by episode
def split_data_by_episodes(samples, horizon, n_obs=0, n_acs=0, truncate_size=-1):
    assert horizon >= 1

    # new change
    has_latent = True
    if 'latent' not in samples:
        logger.warn("No latent in dataset!")
        has_latent = False

    # how many time steps to ignore in the beginning when the history is not long enough
    pre_remove = 0
    # pre_remove = max(nobs, nacs)

    obs_all = samples['obs']
    acs_all = samples['acs']
    if has_latent:
        latent_all = samples['latent']

    mu_obs = np.mean(obs_all, axis=0)
    sigma_obs = np.std(obs_all, axis=0)

    if 'episode_sizes' in samples:
        episode_sizes = samples['episode_sizes'].flatten().astype(int)
    else:
        episode_sizes = np.array([obs_all.shape[0]])

    ep_sizes = np.cumsum(episode_sizes)

    # truncate
    if 0 < truncate_size < np.sum(episode_sizes):
        ep_max = ep_sizes.size
        for i in range(ep_sizes.size):
            if ep_sizes[i] > truncate_size:
                ep_max = i
                break

        print("Truncating from %d to %d episodes (%d to %d samples)" % (
            ep_sizes.size, ep_max, np.sum(episode_sizes), np.sum(episode_sizes[:, :ep_max])))
        episode_sizes = episode_sizes[:, :ep_max]
        ep_sizes = ep_sizes[:ep_max]

    obs_list = np.split(obs_all, ep_sizes, axis=0)
    acs_list = np.split(acs_all, ep_sizes, axis=0)
    if has_latent:
        latent_list = np.split(latent_all, ep_sizes, axis=0)

    obs_start_list = []
    acs_start_list = []
    next_obs_list = []
    latent_start_list = []
    prev_obs_start_list = []
    prev_acs_start_list = []
    obs_seq_list = []
    ac_seq_list = []
    done_list = []

    new_episode_sizes = []

    for ep in range(ep_sizes.size):
        obs = obs_list[ep]
        acs = acs_list[ep]
        if has_latent:
            latent = latent_list[ep]

        ### Action sequences

        # row i corresponds to actions taken until action i
        prev_acs = split_dim_np(fill_n_prev(acs, horizon - 1), axis=1, new_shape=[horizon - 1] + list(acs.shape[1:]))
        prev_obs_horizon = split_dim_np(fill_n_prev(obs, horizon, initial_zero=False),
                                        axis=1, new_shape=[horizon] + list(obs.shape[1:]))

        # appending to create action sequence list removing the first plan_hor-1 elements
        #  (and last element since we don't know the result of it)
        ac_seq = np.concatenate([acs[:, None], prev_acs], axis=1)[horizon - 1:-1]

        # un-reversing and removing the initial ones
        ac_seq = ac_seq[pre_remove:, ::-1, :]

        ### Observation sequences

        # obs sequence in the future
        obs_seq = np.concatenate([obs[:, None], prev_obs_horizon], axis=1)[horizon:]
        # un reversing
        obs_seq = obs_seq[pre_remove:, ::-1, :]

        # trashing data
        if obs_seq.shape[0] < 2:
            print("[] Trashing rollout %d due to lack of samples" % int(ep))
            continue

        ### Action histories

        prev_acs = split_dim_np(fill_n_prev(acs, n_acs), axis=1, new_shape=[n_acs] + list(acs.shape[1:]))
        acs_start = acs[pre_remove:-horizon]
        prev_acs_start = prev_acs[pre_remove:-horizon]

        ### Observation histories

        # we don't use initial_zero=False here bc we remove the first pre_remove anyways
        prev_obs = split_dim_np(fill_n_prev(obs, n_obs), axis=1, new_shape=[n_obs] + list(obs.shape[1:]))
        obs_start = obs[pre_remove:-horizon]
        next_obs = obs[pre_remove + 1:-horizon + 1]
        if has_latent:
            latent_start = latent[pre_remove:-horizon].astype(int)
        else:
            latent_start = np.zeros((obs_start.shape[0], 0)).astype(int)
        prev_obs_start = prev_obs[pre_remove:-horizon]

        obs_seq_list.append(obs_seq)  # (N x H+1 x dO)
        ac_seq_list.append(ac_seq)  # (N x H x dU)
        obs_start_list.append(obs_start)  # (N x dO)
        acs_start_list.append(acs_start)  # (N x dU)
        next_obs_list.append(next_obs)  # (N x dO)
        latent_start_list.append(latent_start)  # (N x 1)
        prev_obs_start_list.append(prev_obs_start)  # (N x nobs x dO)
        prev_acs_start_list.append(prev_acs_start)  # (N x nacs x dO)
        done_list.append(np.array([False for _ in range(obs_start.shape[0] - 1)] + [True], dtype=np.bool))
        new_episode_sizes.append(obs_start.shape[0])

    # remove bad eps:
    episode_sizes = np.array(new_episode_sizes)

    # some input statistics
    delta_obs = np.concatenate(next_obs_list, axis=0) - np.concatenate(obs_start_list, axis=0)
    mu_delta_obs = np.mean(delta_obs, axis=0)
    sigma_delta_obs = np.std(delta_obs, axis=0)
    next_obs_sigma_list = [np.tile(sigma_delta_obs[None], (next_obs.shape[0], 1)) for next_obs in next_obs_list]

    return_dict = {
        'mu_obs': mu_obs,
        'sigma_obs': sigma_obs,
        'mu_delta_obs': mu_delta_obs,
        'sigma_delta_obs': sigma_delta_obs,
        'episode_sizes': episode_sizes,
        'done': done_list,
        'obs_full': obs_list,
        'act_full': acs_list,
        # 'obs_seq': obs_seq_list,
        'latent': latent_start_list,
        'act_seq': ac_seq_list,
        'obs': obs_start_list,
        'act': acs_start_list,
        'prev_obs': prev_obs_start_list,
        'prev_act': prev_acs_start_list,
        'next_obs': next_obs_list,
        'next_obs_sigma': next_obs_sigma_list,
        'goal_obs': obs_seq_list,
    }

    return return_dict
