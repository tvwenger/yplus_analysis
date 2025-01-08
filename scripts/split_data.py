import os
import sys
import pickle

datadir = sys.argv[1]

fnames = [
    # f"{datadir}/hii_noise.pickle",
    # f"{datadir}/hii_intensity-noise.pickle",
    # f"{datadir}/dig_noise_1.0.pickle",
    f"{datadir}/dig_intensity-noise_1.0.pickle",
]

for fname in fnames:
    dirname = fname.replace(".pickle", "")
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    with open(fname, "rb") as f:
        data = pickle.load(f)
    for idx, row in data.iterrows():
        print(fname, idx)
        with open(f"{dirname}/{idx}.pickle", "wb") as f:
            pickle.dump(row, f)
