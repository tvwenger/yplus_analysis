import os
import sys
import pickle

import numpy as np

import pymc as pm
import bayes_spec
import bayes_yplus

from bayes_spec import SpecData, Optimize
from bayes_yplus import YPlusModel

def main(dirname, idx):
    print(f"Starting job on idx = {idx}")
    print(f"pymc version: {pm.__version__}")
    print(f"bayes_spec version: {bayes_spec.__version__}")
    print(f"bayes_yplus version: {bayes_yplus.__version__}")
    result = {
        "idx": idx,
        "exception": "",
        "results": {},
    }

    # load data
    with open(f"data/{dirname}/{idx:06d}.pickle", "rb") as f:
        datum = pickle.load(f)

    # get data
    channel = np.linspace(-300.0, 300.0, 1201)
    spectrum = 1000.0 * np.array(datum['Avg_Spectra']) # mK
    if np.all(np.isnan(spectrum)) or np.all(spectrum == 0.0):
        result["exception"] = "no data"
        return result

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

    # skip if there does not appear to be any signal
    #if not np.any(spectrum > 2.5*rms):
    #    result["exception"] = "no apparent signal"
    #    return result
    
    # save
    observation = SpecData(
        channel,
        spectrum,
        rms,
        xlabel=r"$V_{\rm LSR}$ (km s$^{-1}$)",
        ylabel=r"$T_B$ (mK)",
    )
    data = {"observation": observation}

    try:
        # Initialize optimizer
        opt = Optimize(
            YPlusModel,
            data,
            max_n_clouds=5,
            baseline_degree=2,
            seed=1234,
            verbose=True,
        )
        opt.add_priors(
            prior_H_area = 1000.0,
            prior_H_center = [0.0, 25.0],
            prior_H_fwhm = 20.0,
            prior_He_H_fwhm_ratio = 0.1,
            prior_yplus = 0.1,
            prior_rms = 10.0,
            prior_baseline_coeffs = [1.0, 1.0, 0.5],
            ordered = False,
        )
        opt.add_likelihood()
        fit_kwargs = {
            "rel_tolerance": 0.01,
            "abs_tolerance": 0.1,
            "learning_rate": 1e-2,
        }
        sample_kwargs = {
            "chains": 4,
            "cores": 4,
            "init_kwargs": fit_kwargs,
            "nuts_kwargs": {"target_accept": 0.8},
        }
        opt.optimize(bic_threshold=10.0, sample_kwargs=sample_kwargs, fit_kwargs=fit_kwargs, approx=False)

        # save BICs and results for each model
        results = {0: {"bic": opt.best_model.null_bic()}}
        for n_gauss, model in opt.models.items():
            results[n_gauss] = {}
            if len(model.solutions) > 1:
                results[n_gauss]["exception"] = "multiple solutions"
            elif len(model.solutions) == 1:
                results[n_gauss]["bic"] = model.bic(solution=0)
                results[n_gauss]["summary"] = pm.summary(model.trace.solution_0)
            else:
                results[n_gauss]["exception"] = "no solution"
        result["results"] = results
        return result

    except Exception as ex:
        result["exception"] = ex
        return result


if __name__ == "__main__":
    dirname = sys.argv[1]
    idx = int(sys.argv[2])
    output = main(dirname, idx)
    if output["exception"] != "":
        print(output["exception"])

    # save results
    outdirname = f"results/{dirname}_results"
    if not os.path.isdir(outdirname):
        os.mkdir(outdirname)
    fname = f"{outdirname}/{idx:06d}.pkl"
    with open(fname, "wb") as f:
        pickle.dump(output, f)
