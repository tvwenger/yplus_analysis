"""
fit.py
Run bayes_yplus on single GDIGS spectrum
Trey V. Wenger - 2024-2025
"""

import sys
import pickle

import numpy as np

import pymc as pm
import bayes_spec
import bayes_yplus

from bayes_spec import SpecData, Optimize
from bayes_yplus import YPlusModel


def main(idx, infile):
    print(f"Starting job on {infile}")
    print(f"pymc version: {pm.__version__}")
    print(f"bayes_spec version: {bayes_spec.__version__}")
    print(f"bayes_yplus version: {bayes_yplus.__version__}")
    result = {
        "idx": idx,
        "exception": "",
        "results": {},
    }

    # load data
    with open(infile, "rb") as f:
        datum = pickle.load(f)

    # get data
    channel = np.linspace(-300.0, 300.0, 1201)
    spectrum = 1000.0 * np.array(datum["Avg_Spectra"])  # mK
    if np.all(np.isnan(spectrum)) or np.all(spectrum == 0.0):
        result["exception"] = "no data"
        return result

    # estimate rms from line-free channels
    meds = np.array(
        [
            np.median(spectrum[channel < -150.0]),
            np.median(spectrum[channel > 150.0]),
        ]
    )
    rmss = np.array(
        [
            1.4826 * np.median(np.abs(spectrum[channel < -150.0] - meds[0])),
            1.4826 * np.median(np.abs(spectrum[channel > 150.0] - meds[1])),
        ]
    )
    rms = np.mean(rmss)

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
            baseline_degree=1,
            seed=1234,
            verbose=True,
        )
        opt.add_priors(
            prior_H_area=1000.0,  # width of the H_area prior (mK km s-1)
            prior_H_center=[0.0, 25.0],  # mean and width of H_center prior (km s-1)
            prior_H_fwhm=[25.0, 10.0],  # mean and width of H FWHM prior (km s-1)
            prior_He_H_fwhm_ratio=[1.0, 0.1],  # mean and width of He/H FWHM ratio prior
            prior_yplus=0.05,  # width of yplus prior
            prior_fwhm_L=50.0,  # width of Lorentzian FWHM prior (km s-1)
            prior_rms=None,  # do not infer spectral rms
            prior_baseline_coeffs=None,  # use default baseline priors
            ordered=False,  # do not assume ordered components
        )
        opt.add_likelihood()
        fit_kwargs = {
            "rel_tolerance": 0.01,
            "abs_tolerance": 0.05,
            "learning_rate": 1e-2,
        }
        sample_kwargs = {
            "chains": 8,
            "cores": 8,
            "tune": 1000,
            "draws": 1000,
            "init_kwargs": fit_kwargs,
            "nuts_kwargs": {"target_accept": 0.8},
        }
        opt.optimize(bic_threshold=10.0, sample_kwargs=sample_kwargs, fit_kwargs=fit_kwargs, approx=False)

        # save BICs and results for each model
        results = {0: {"bic": opt.models[1].null_bic()}}
        for n_gauss, model in opt.models.items():
            results[n_gauss] = {"bic": np.inf, "solutions": {}}
            for solution in model.solutions:
                # get BIC
                bic = model.bic(solution=solution)

                # get summary
                summary = pm.summary(model.trace[f"solution_{solution}"], round_to=None)

                # check convergence
                converged = summary["r_hat"].max() < 1.05

                if converged and bic < results[n_gauss]["bic"]:
                    results[n_gauss]["bic"] = bic

                # save posterior samples for un-normalized params (except baseline)
                data_vars = list(model.trace[f"solution_{solution}"].data_vars)
                data_vars = [data_var for data_var in data_vars if ("baseline" in data_var) or not ("norm" in data_var)]

                # only save posterior samples if converged
                results[n_gauss]["solutions"][solution] = {
                    "bic": bic,
                    "summary": summary,
                    "converged": converged,
                    "trace": (
                        model.trace[f"solution_{solution}"][data_vars].sel(draw=slice(None, None, 10))
                        if converged
                        else None
                    ),
                }
        result["results"] = results
        return result

    except Exception as ex:
        result["exception"] = ex
        return result


if __name__ == "__main__":
    dirname = sys.argv[1]
    idx = int(sys.argv[2])
    htc_type = sys.argv[3]

    if htc_type == "slurm":
        infile = f"data/{dirname}/{idx}.pkl"
        outfile = f"results/{dirname}/{idx}_bayes_yplus.pkl"
    else:
        infile = f"{idx}.pkl"
        outfile = f"{idx}_bayes_yplus.pkl"

    output = main(idx, infile)
    if output["exception"] != "":
        print(output["exception"])

    # save results
    with open(outfile, "wb") as f:
        pickle.dump(output, f)
