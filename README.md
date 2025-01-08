# yplus_analysis
Run `bayes_yplus` on GDIGS data with SLURM or Condor.

## Installation
```bash
git clone git@github.com:tvwenger/yplus_analysis.git
cd yplus_analysis
mkdir -p data/ results/ logs/
conda env create -f environment.yml
conda activate bayes_yplus
```

## Docker
The `bayes_yplus` environment is also provided via a `docker` container.
```
docker build -t tvwenger/bayes_yplus:v1.3.1 .
docker push tvwenger/bayes_yplus:v1.3.1
```

This `docker` container can be [converted to an Apptainer image](https://chtc.cs.wisc.edu/uw-research-computing/htc-docker-to-apptainer) on CHTC:
```
condor_submit -i condor/build.sub
apptainer build /staging/tvwenger/bayes_yplus-v1.3.1.sif docker://tvwenger/bayes_yplus:v1.3.1
```

## First steps
First, we split up the data into individual data files for parallel processing. Copy the data files into `data/` first, then:
```
cd scripts/
python split_data.py
```
Individual data files are stored in `data/<datadir>/` where `<datadir>` is like `hii_noise`, `hii_intensity-noise`, `dig_noise_1.0`, `dig_intensity-noise_1.0`

## Parallel processing with SLURM
The script `fit.py` runs `bayes_yplus`'s optimization algorithm on a single pickle file. With SLURM, the data are assumed to be in `data/<datadir>/` and the results go in `results/<datadir>/` as individual pickle files.
```
python fit_G049.py <idx> slurm
```

The script `submit_G049.sh` is a SLURM script that we use to analyze each pixel in parallel. SLURM logs are written to `logs/`.
```
sbatch submit_G049.sh
```