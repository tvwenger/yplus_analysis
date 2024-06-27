import sys
import os
import pickle
import dill
import arviz as az
import numpy as np

datasets = [
    "hii_noise",
    "hii_intensity-noise",
    "dig_noise_1.0",
    "dig_intensity-noise_1.0",
]

idx = int(sys.argv[1])
datasets = datasets[idx:idx+1]

for dataset in datasets:
    print(dataset)
    fname = f"data/{dataset}.pickle"
    with open(fname, 'rb') as f:
        data = pickle.load(f)

    # storage for MCMC
    data["n_gauss MCMC"] = 0
    data["H Height MCMC"] = None
    data["H Velocity MCMC"] = None
    data["H FWHM MCMC"] = None
    data["He Height MCMC"] = None
    data["He Velocity MCMC"] = None
    data["He FWHM MCMC"] = None
    data["He/H FWHM Ratio MCMC"] = None
    data["yplus MCMC"] = None
    data["coeffs MCMC"] = None
    data["err H Height MCMC"] = None
    data["err H Velocity MCMC"] = None
    data["err H FWHM MCMC"] = None
    data["err He Height MCMC"] = None
    data["err He Velocity MCMC"] = None
    data["err He FWHM MCMC"] = None
    data["err He/H FWHM Ratio MCMC"] = None
    data["err yplus MCMC"] = None
    data["err coeffs MCMC"] = None
    data["good MCMC"] = False
    data["exception MCMC"] = ""
    data["n_gauss BIC"] = None

    for idx, row in data.iterrows():
        print(f"{dataset} {idx}")

        # load result
        resultfname = f"results/{dataset}_results/{idx:06d}.pkl"
        if not os.path.exists(resultfname):
            data.at[idx, "exception MCMC"] = "no result"
            continue
        with open(resultfname, "rb") as f:
            result = dill.load(f)

        # catch failures
        if "exception" in result.keys():
            data.at[idx, "exception MCMC"] = result["exception"]
            continue

        # catch no solution
        if len(result["best_model"].solutions) == 0:
            data.at[idx, "exception MCMC"] = "no solution"
            continue

        # catch multiple solutions
        if len(result["best_model"].solutions) > 1:
            data.at[idx, "exception MCMC"] = "multiple solutions"

        # check null BIC
        bic = result["best_model"].bic(solution=0)
        bic_threshold = 10.0
        if result["best_model"].null_bic()+bic_threshold < bic:
            continue

        # save result
        n_gauss = result["best_model"].n_clouds
        baseline_degree = result["best_model"].baseline_degree
        bics = np.array([result["bics"][i] for i in range(len(result["bics"]))])
        data.at[idx, "n_gauss BIC"] = bics
        point_stats = az.summary(
            result["best_model"].trace.solution_0,
            kind='stats',
            hdi_prob=0.68
        )
        
        data.at[idx, "good MCMC"] = True
        data.at[idx, "n_gauss MCMC"] = n_gauss
        data.at[idx, "H Height MCMC"] = np.array([point_stats['mean'][f"H_amplitude[{i}]"] for i in range(n_gauss)])
        data.at[idx, "H Velocity MCMC"] = np.array([point_stats['mean'][f"H_center[{i}]"] for i in range(n_gauss)])
        data.at[idx, "H FWHM MCMC"] = np.array([point_stats['mean'][f"H_fwhm[{i}]"] for i in range(n_gauss)])
        data.at[idx, "He Height MCMC"] = np.array([point_stats['mean'][f"He_amplitude[{i}]"] for i in range(n_gauss)])
        data.at[idx, "He Velocity MCMC"] = np.array([point_stats['mean'][f"He_center[{i}]"] for i in range(n_gauss)])
        data.at[idx, "He FWHM MCMC"] = np.array([point_stats['mean'][f"He_fwhm[{i}]"] for i in range(n_gauss)])
        data.at[idx, "He/H FWHM Ratio MCMC"] = point_stats['mean']["He_H_fwhm_ratio"]
        data.at[idx, "yplus MCMC"] = point_stats['mean']["yplus"]
        data.at[idx, "coeffs MCMC"] = np.array([point_stats['mean'][f"coeffs[{i}]"] for i in range(baseline_degree+1)])
        data.at[idx, "err H Height MCMC"] = np.array([point_stats['sd'][f"H_amplitude[{i}]"] for i in range(n_gauss)])
        data.at[idx, "err H Velocity MCMC"] = np.array([point_stats['sd'][f"H_center[{i}]"] for i in range(n_gauss)])
        data.at[idx, "err H FWHM MCMC"] = np.array([point_stats['sd'][f"H_fwhm[{i}]"] for i in range(n_gauss)])
        data.at[idx, "err He Height MCMC"] = np.array([point_stats['sd'][f"He_amplitude[{i}]"] for i in range(n_gauss)])
        data.at[idx, "err He Velocity MCMC"] = np.array([point_stats['sd'][f"He_center[{i}]"] for i in range(n_gauss)])
        data.at[idx, "err He FWHM MCMC"] = np.array([point_stats['sd'][f"He_fwhm[{i}]"] for i in range(n_gauss)])
        data.at[idx, "err He/H FWHM Ratio MCMC"] = point_stats['sd']["He_H_fwhm_ratio"]
        data.at[idx, "err yplus MCMC"] = point_stats['sd']["yplus"]
        data.at[idx, "err coeffs MCMC"] = np.array([point_stats['sd'][f"coeffs[{i}]"] for i in range(baseline_degree+1)])
        
    outfname = f"results/{dataset}-mcmc.pickle"
    with open(outfname, "wb") as f:
        pickle.dump(data, f)
