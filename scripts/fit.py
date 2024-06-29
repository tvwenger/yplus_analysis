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
    meds = np.array([
        np.median(spectrum[channel < -150.0]),
        np.median(spectrum[channel > 150.0]),
    ])
    rmss = np.array([
        1.4826 * np.median(np.abs(spectrum[channel < -150.0] - meds[0])),
        1.4826 * np.median(np.abs(spectrum[channel > 150.0] - meds[1])),
    ])
    rms = np.mean(rmss)
    prior_H_amplitude = np.percentile(spectrum, 99.0)

    # skip if there does not appear to be any signal
    if not np.any(spectrum > 3.0*rms):
        return {"idx": idx, "exception": "no apparent signal", "spec_rms": rms}

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
            "n_init": 20_000,
            "init_kwargs": fit_kwargs,
            "nuts_kwargs": {"target_accept": 0.8},
        }
        opt.optimize(sample_kwargs=sample_kwargs, bic_threshold=10.0, approx=False)

        # get BICs for each model
        bics = {0: opt.best_model.null_bic()}
        for n_gauss, model in opt.models.items():
            if len(model.solutions) > 0:
                bics[n_gauss] = model.bic(solution=0)
            else:
                bics[n_gauss] = np.inf
        result = {"idx": idx, "best_model": opt.best_model, "bics": bics, "spec_rms": rms}
        return result

    except Exception as ex:
        return {"idx": idx, "exception": ex, "spec_rms": rms}


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

