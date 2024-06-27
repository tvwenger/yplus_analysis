import os
import sys
import pickle
import dill
import numpy as np
from mcmc_yplus.optimize import Optimize


def main(dirname, idx):
    print(f"Starting job on idx = {idx}")

    # load data
    with open(f"data/{dirname}/{idx:06d}.pickle", "rb") as f:
        datum = pickle.load(f)

    # get data
    channel = np.linspace(-300.0, 300.0, 1201)
    spectrum = np.array(datum['Avg_Spectra'])
    if np.all(np.isnan(spectrum)) or np.all(spectrum == 0.0):
        return {"idx": idx, "exception": "no data"}

    # estimate rms from line-free channels
    good = (channel < -150.0) + (channel > 150.0)
    med = np.median(spectrum[good])
    rms = 1.4826 * np.median(np.abs(spectrum[good] - med))
    prior_H_amplitude = np.percentile(spectrum, 99.0)

    # save
    data = {
        "velocity": channel,
        "spectrum": spectrum,
        "noise": np.ones_like(channel)*rms,
    }

    try:
        # Initialize optimizer
        opt = Optimize(
            data,
            max_n_clouds=5,
            baseline_degree=3,
            seed=1234,
            verbose=True,
        )
        opt.set_priors(
            prior_H_amplitude=prior_H_amplitude,
            prior_H_center=[0.0, 30.0],
            prior_H_fwhm=20.0,
            prior_yplus=0.1,
            prior_He_H_fwhm_ratio=0.1,
        )
        fit_kwargs = {
            "learning_rate": 1e-2,
            "abs_tolerance": 0.03,
            "rel_tolerance": 0.03,
        }
        sample_kwargs = {
            "chains": 4,
            "cores": 4,
            "n_init": 100_000,
            "init_kwargs": fit_kwargs,
            "nuts_kwargs": {"target_accept": 0.8},
        }
        opt.optimize(sample_kwargs=sample_kwargs, fit_kwargs=fit_kwargs, bic_threshold=10.0)

        # get BICs for each model
        bics = {0: opt.best_model.null_bic()}
        for n_gauss, model in opt.models.items():
            if len(model.solutions) > 0:
                bics[n_gauss] = model.bic(solution=0)
            else:
                bics[n_gauss] = np.inf
        result = {"idx": idx, "best_model": opt.best_model, "bics": bics}
        return result

    except Exception as ex:
        return {"idx": idx, "exception": ex}


if __name__ == "__main__":
    dirname = sys.argv[1]
    idx = int(sys.argv[2])
    output = main(dirname, idx)
    if "exception" in output.keys():
        print(output["exception"])

    # save results
    outdirname = f"results/{dirname}_results"
    if not os.path.isdir(outdirname):
        os.mkdir(outdirname)
    fname = f"{outdirname}/{idx:06d}.pkl"
    with open(fname, "wb") as f:
        dill.dump(output, f)

