import sys
import os
import pickle
import numpy as np
from numpy.polynomial import Polynomial
import pandas as pd


def gaussian(x, amp, fwhm, center):
    return amp * np.exp(-4.0 * np.log(2.0) * (x - center) ** 2.0 / fwhm**2.0)


def get_best_model(result, bic_threshold=10.0):
    """Determine the best bayes_yplus result, and return the best solution."""
    # keep only best model
    best_bic = np.inf
    best_n_gauss = 0
    best_solution = 0
    best_num_solutions = 0

    for n_gauss in result["results"].keys():
        this_bic = np.inf
        this_solution = None
        this_num_solutions = 0
        if "bic" in result["results"][n_gauss]:
            this_bic = result["results"][n_gauss]["bic"]

        if "solutions" in result["results"][n_gauss].keys():
            this_num_solutions = len(result["results"][n_gauss]["solutions"])
            for solution in result["results"][n_gauss]["solutions"].keys():
                converged = result["results"][n_gauss]["solutions"][solution]["converged"]
                bic = result["results"][n_gauss]["solutions"][solution]["bic"]

                # check if this solution is (1) converged and (2) has a better BIC
                if converged and bic <= this_bic:
                    this_bic = bic
                    this_solution = solution

        # see if the best solution from this n_gauss has a better BIC
        if np.isinf(best_bic) or this_bic < (best_bic - bic_threshold):
            best_bic = this_bic
            best_n_gauss = n_gauss
            best_solution = this_solution
            best_num_solutions = this_num_solutions

    # save result
    if best_n_gauss in result["results"].keys():
        result["results"] = result["results"][best_n_gauss]
        if "solutions" in result["results"].keys() and best_solution is not None:
            result["results"] = result["results"]["solutions"][best_solution]
            return result, best_bic, best_n_gauss, best_num_solutions

    # no result
    return None, best_bic, best_n_gauss, best_num_solutions


def main(dataset, bic_threshold=10.0):
    # channel definition (VLSR, km/s)
    chan = np.linspace(-300.0, 300.0, 1201)
    chan_norm = 2.0 * (chan - chan.mean()) / np.ptp(chan)

    # load original dataset
    fname = f"data/{dataset}.pickle"
    with open(fname, "rb") as f:
        data = pickle.load(f)

    # prepare results
    sightline_results = []
    cloud_results = []

    # loop over data
    for idx, datum in data.iterrows():
        print(f"{dataset} {idx}", end="\r")

        # get data
        channel = np.linspace(-300.0, 300.0, 1201)
        spectrum = 1000.0 * np.array(datum["Avg_Spectra"])  # mK
        if np.all(np.isnan(spectrum)) or np.all(spectrum == 0.0):
            # no data
            continue

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
        spec_median = np.median(spectrum)
        spec_rms = np.mean(rmss)

        sightline_result = {
            "sightline_idx": idx,
            "glong": datum["GLong"],
            "glat": datum["GLat"],
            "exception": None,
            "best_n_gauss": 0,
            "best_bic": np.inf,
            "best_num_solutions": 0,
            "spec_rms": spec_rms,
            "spectrum": list(1000.0 * np.array(data.loc[idx, "Avg_Spectra"]).astype(float)),
            "model": list(np.zeros_like(datum["Avg_Spectra"])),
        }

        # load bayes_yplus output
        outputfname = f"results/{dataset}/{idx}_bayes_yplus.pkl"
        if not os.path.exists(outputfname):
            sightline_result["exception"] = "no output"
            sightline_results.append(sightline_result)
            continue
        with open(outputfname, "rb") as f:
            output = pickle.load(f)

        # get best model
        best_model, best_bic, best_n_gauss, best_num_solutions = get_best_model(output, bic_threshold=bic_threshold)
        sightline_result["best_bic"] = best_bic
        sightline_result["best_n_gauss"] = best_n_gauss
        sightline_result["best_num_solutions"] = best_num_solutions

        if best_model is None:
            sightline_result["exception"] = "no model"
            sightline_results.append(sightline_result)
            continue

        # add model baseline parameters to sightline results
        _BASELINE_DEGREE = 2
        _STATS = ["mean", "sd", "hdi_3%", "hdi_97%"]
        for key in [f"baseline_observation_norm[{i}]" for i in range(_BASELINE_DEGREE + 1)]:
            for stat in _STATS:
                sightline_result[f"{key}_{stat}"] = best_model["results"]["summary"][stat][key]

        # evaluate baseline model
        coeffs = np.array(
            [
                best_model["results"]["summary"]["mean"][f"baseline_observation_norm[{i}]"] / (i + 1.0) ** i
                for i in range(_BASELINE_DEGREE + 1)
            ]
        )
        baseline = Polynomial(coeffs)(chan_norm) * spec_rms + spec_median
        model = baseline.copy()

        # loop over clouds
        for cloud in range(best_n_gauss):
            # evaluate model for this cloud
            cloud_H_model = gaussian(
                chan,
                best_model["results"]["summary"]["mean"][f"H_amplitude[{cloud}]"],
                best_model["results"]["summary"]["mean"][f"H_fwhm[{cloud}]"],
                best_model["results"]["summary"]["mean"][f"H_center[{cloud}]"],
            )
            cloud_He_model = gaussian(
                chan,
                best_model["results"]["summary"]["mean"][f"He_amplitude[{cloud}]"],
                best_model["results"]["summary"]["mean"][f"He_fwhm[{cloud}]"],
                best_model["results"]["summary"]["mean"][f"He_center[{cloud}]"],
            )
            cloud_model = baseline + cloud_H_model + cloud_He_model
            model += cloud_H_model + cloud_He_model

            cloud_result = {
                "sightline_idx": idx,
                "cloud": cloud,
                "glong": datum["GLong"],
                "glat": datum["GLat"],
                "spec_rms": spec_rms,
                "model": list(cloud_model),
            }

            # add model cloud parameters to cloud results
            for key in ["H_amplitude", "H_fwhm", "H_center", "He_amplitude", "He_fwhm", "yplus"]:
                for stat in _STATS:
                    cloud_result[f"{key}_{stat}"] = best_model["results"]["summary"][stat][f"{key}[{cloud}]"]

            cloud_results.append(cloud_result)

        # save model
        sightline_result["model"] = list(model)
        sightline_results.append(sightline_result)

    # write to data files
    with open(f"results/{dataset}-bayes_yplus-sightline_results.pkl", "wb") as f:
        pickle.dump(sightline_results, f)
    with open(f"results/{dataset}-bayes_yplus-cloud_results.pkl", "wb") as f:
        pickle.dump(cloud_results, f)

    # save parameters to CSV
    sightline_data = pd.DataFrame(sightline_results)
    all_keys = sightline_data.keys()
    keep_keys = [key for key in all_keys if key not in ["spectrum", "model"]]
    sightline_data.astype({"spectrum": str, "model": str}).to_csv(
        f"results/{dataset}-bayes_yplus-sightline_results.csv",
        columns=keep_keys,
        index=False,
    )

    cloud_data = pd.DataFrame(cloud_results)
    all_keys = cloud_data.keys()
    keep_keys = [key for key in all_keys if key not in ["model"]]
    cloud_data.astype({"model": str}).to_csv(
        f"results/{dataset}-bayes_yplus-cloud_results.csv",
        columns=keep_keys,
        index=True,
        index_label="cloud_idx",
    )


if __name__ == "__main__":
    main(sys.argv[1], bic_threshold=10.0)
