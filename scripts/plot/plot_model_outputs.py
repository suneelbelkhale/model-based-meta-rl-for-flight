import argparse
import os

import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.io import loadmat

from mbrl.experiments.file_manager import FileManager
from mbrl.utils.file_utils import import_config


def get_rectangle(px, py, area=0):
    if area < 1e-8:
        area = 0
    side = np.sqrt(area)
    rect = np.array([
        [side / 2, side / 2],
        [side / 2, -side / 2],
        [-side / 2, -side / 2],
        [-side / 2, side / 2],
        [side / 2, side / 2]
    ])

    # print(px, py, area)

    return rect + np.array([px, py])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--evalfile', type=str, default="model_outputs_holdout.mat")
    parser.add_argument('--save', action='store_true', help="save the plots to the experiment folder")
    parser.add_argument('-k', type=int, required=True, help='Number of plots to create from file')
    parser.add_argument('-sp', '--saveprefix', type=str, default='model_output_plot_', help='Prefix to prepend to output plots')
    parser.add_argument('-H', type=int, help='horizon to truncate rollouts to', default=0)
    parser.add_argument('-hd', '--hide', action='store_true', help='hide plots')
    parser.add_argument('-sa', '--sample_sorted', action='store_true',
                        help='sample k evenly across worst to best rollouts to visualize, otherwise use k best')

    args = parser.parse_args()
    #
    # if args.savefolder is not '' and not os.path.exists(args.savefolder):
    #     raise Exception("Save Directory %s does not exist!" % args.savefolder)

    config_fname = os.path.abspath(args.config)
    assert os.path.exists(config_fname), 'Config: {0} does not exist'.format(config_fname)
    params = import_config(config_fname)
    params.freeze()
    file_manager = FileManager(params.exp_name, is_continue=True)

    save_folder = None
    if args.save:
        save_folder = os.path.join(file_manager.exp_dir, 'plots')
        os.makedirs(save_folder, exist_ok=True)

    def size_fit(arr):
        assert len(arr.shape) >= 3
        arr = np.expand_dims(arr, axis=-2)
        # num_points x H+1 x 1 x odim
        if len(arr.shape) == 4:
            out = np.expand_dims(arr, axis=-2)

        # num_points x NP x H+1 x 1 x odim
        elif len(arr.shape) == 5:
            out = np.transpose(arr, axes=(0, 2, 3, 1, 4))

        else:
            raise NotImplementedError

        return out

    dct = loadmat(os.path.join(file_manager.exp_dir, args.evalfile))
    cost = dct['costs'][0]
    traj = size_fit(dct['obs'])
    ac_seq = dct['act']
    obs_seq = size_fit(dct['goal_obs'])

    # (N, H, NMODELS, NPART, OBDIM)
    assert len(traj.shape) == 5
    # mpc specific things
    N = traj.shape[0]
    H = traj.shape[1]  # H + 1 technically
    nn = traj.shape[2]
    npart = traj.shape[3]
    dO = traj.shape[4]

    start_obs = traj[:, 0, 0, 0]
    mean = np.mean(start_obs, axis=0)
    std = np.std(start_obs, axis=0)

    assert obs_seq.shape[0] == N
    assert ac_seq.shape[0] == N

    if args.H > 0:
        assert args.H <= H - 1
        traj = traj[:, :args.H + 1]
        obs_seq = obs_seq[:, :(args.H + 1) * dO]
        H = args.H + 1

    traj_by_mp = np.transpose(traj, (2, 3, 0, 1, 4))  # (NM, NP, N, H, dO)

    obs_seq_exp = obs_seq.reshape(N, H, dO)
    dists = (traj_by_mp - obs_seq_exp)  # (NM, NP, N, H, dO)
    weighted_dist = np.multiply(dists, 1 / std)  # weights is (d0,) (1 / stdev)

    # L2 loss
    L2 = np.mean(np.sqrt(np.sum(np.square(weighted_dist), axis=4)), axis=3)  # (NM, NP, N)

    # metric 1: mean accuracy among all models/parts
    # metric 2: accuracy of closest among all models/parts
    # metric 3: accuracy of closest particle avg among all models

    L2_N = np.transpose(L2, (2, 0, 1))

    overall_mean = np.mean(L2_N)
    unweighted_mean = np.mean(np.sqrt(np.sum(np.square(dists), axis=(3, 4))))
    unweighted_pixel_mean = np.mean(np.sqrt(np.sum(np.square(dists[:, :, :, :, :2]), axis=(3, 4))))
    print("Overall Mean Cost (L2):", overall_mean, "Unweighted Mean Cost (L2):", unweighted_mean,
          "Unweighted Pixel Mean Cost (L2):", unweighted_pixel_mean)

    metric1 = np.mean(L2_N, axis=(1, 2))
    metric2 = np.amin(L2_N, axis=(1, 2))
    metric3 = np.amin(np.mean(L2_N, axis=2), axis=1)

    metrics = [metric1, metric2, metric3]
    mnames = ['Min Mean Distance', 'Min Distance', 'Min Mean Distance over Particles']
    gspec = [3, len(metrics)]
    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1. * i / nn) for i in range(nn)]

    if args.sample_sorted:
        # sample evenly across sorted list
        idxs = np.arange(args.k) * (N // args.k)
        argmins = [np.argsort(m)[idxs] for m in metrics]
    else:
        argmins = [np.argpartition(m, args.k)[:args.k] for m in metrics]

    for k in range(args.k):
        f = plt.figure(constrained_layout=True, figsize=(18, 12))
        title = 'K = %d, H = %d, Pixel Mean Cost = %.8f' % (k, H - 1, unweighted_pixel_mean)
        if args.sample_sorted:
            title += ' IDX = %d/%d' % (idxs[k], N)
        f.suptitle(title)
        gs = gridspec.GridSpec(gspec[0], gspec[1], figure=f)

        # plot trajectory k
        for i in range(len(metrics)):
            trajectories = traj_by_mp[:, :, argmins[i][k]]  # (NM, NP, H, dO)
            sample_idxs = argmins[i]
            mname = mnames[i]
            cost = metrics[i][sample_idxs[k]]
            ax = f.add_subplot(gs[:2, i % gspec[1]])
            ax.set_title("%s: %.6f" % (mname, cost))
            ax.set_xlim([0., 1.])
            ax.set_ylim([1., 0.])
            ax.set_aspect('equal')
            ax.set_facecolor('black')

            ax2 = f.add_subplot(gs[2:, i % gspec[1]])

            true_traj = obs_seq_exp[sample_idxs[k]]
            # print(true_traj)
            # import ipdb; ipdb.set_trace()
            for p in range(npart):
                # ax.set_prop_cycle(color=colors)
                for m in range(nn):
                    # plot trajectory
                    p_m_traj = trajectories[m, p]
                    hr = p_m_traj.shape[0]
                    ax.plot(p_m_traj[:, 0], p_m_traj[:, 1], linewidth=0.5, marker='.', color=colors[m], markersize=2)
                    if p_m_traj.shape[1] > 2:
                        ax2.plot(range(hr), p_m_traj[:, 2], marker='.', color=colors[m], label='pr %d,%d' % (p, m))
                    for i in range(1, hr):
                        rc = get_rectangle(*p_m_traj[i])
                        ax.plot(rc[:, 0], rc[:, 1], linewidth=0.5, color=colors[m])

            # ax.set_prop_cycle(None)
            ax.plot(true_traj[:, 0], true_traj[:, 1], color='white', linewidth=1)
            ax.plot(true_traj[:1, 0], true_traj[:1, 1], color='white', marker='x')
            for i in range(1, true_traj.shape[0]):
                rc = get_rectangle(*true_traj[i])
                ax.plot(rc[:, 0], rc[:, 1], linewidth=0.5, color='white')

            if true_traj.shape[1] > 2:
                ax2.plot(range(hr), true_traj[:, 2], color='gray', label='true area')
            ax2.legend()

        if save_folder is not None:
            suff = "H_%d_k_%d.png" % (H - 1, k)
            f.savefig(os.path.join(save_folder, args.saveprefix + suff))

    if not args.hide:
        print("Plotting...")
        plt.show()
        print("Done.")