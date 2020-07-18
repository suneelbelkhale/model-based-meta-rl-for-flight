# takes in a legacy mat file and formats it for current repo
import argparse
import glob
import os

import numpy as np
from scipy.io import loadmat, savemat


def latent_map(fname, ofname, start_idx):
    mat = loadmat(fname)

    if 'latent' not in mat.keys():
        print("Skipping file without latents:", fname)

    latent_all = mat['latent']
    assert latent_all.shape[0] == latent_all.size, "Latent is not 1D for file: " + fname
    latent_all = latent_all.reshape(-1)

    unique = np.unique(latent_all)
    for i, el in enumerate(unique):
        latent_all[latent_all == el] = start_idx + i

    mat['latent'] = latent_all.astype(int).reshape(-1, 1)

    savemat(ofname, mat)
    print("File %s successfully saved to %s with unique elements: %s mapped to %s"
          % (fname, ofname, unique, list(range(start_idx, start_idx + len(unique)))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_value", type=int, required=True, help="the minimum value that gets mapped to")
    parser.add_argument("--input_file", type=str, nargs='+', help="Input mat file(s) to read")
    parser.add_argument("--output_prefix", type=str, default="", help="Prefix for output (default is override)")

    args = parser.parse_args()

    for path in args.input_file:
        output_dir, base = os.path.split(path)
        latent_map(path, os.path.join(output_dir, args.output_prefix + base), args.base_value)
