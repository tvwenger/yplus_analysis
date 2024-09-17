import os
import pickle

fnames = [
    "../data/hii_noise.pickle",
    "../data/hii_intensity-noise.pickle",
    "../data/dig_noise_1.0.pickle",
    "../data/dig_intensity-noise_1.0.pickle",
]

for fname in fnames:
    dirname = fname.replace(".pickle", "")
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    for idx, row in data.iterrows():
        print(fname, idx)
        with open(f"{dirname}/{idx:06d}.pickle", 'wb') as f:
            pickle.dump(row, f)
