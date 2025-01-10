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
docker build -t tvwenger/bayes_yplus:v1.3.2 .
docker push tvwenger/bayes_yplus:v1.3.2
```

N.B. Regarding following Apptainer instructions: there's currently a bug that prevents Apptainer images from being distributed
to jobs when those jobs are submitted as a late marginalization factory. There's also a bug that prevents containers from
accessing DockerHub... But for now we just hope DockerHub works and ignore these Apptainer instructions.

This `docker` container can be [converted to an Apptainer image](https://chtc.cs.wisc.edu/uw-research-computing/htc-docker-to-apptainer) on CHTC:
```
condor_submit -i condor/build.sub
apptainer build /staging/tvwenger/bayes_yplus-v1.3.2.sif docker://tvwenger/bayes_yplus:v1.3.2
```

Or, export the container locally,
```
docker save -o bayes_yplus-v1.3.1.tar tvwenger/bayes_yplus:v1.3.1
# copy bayes_yplus-v1.3.1.tar to CHTC
```
then,
```
condor_submit -i condor/build.sub
apptainer build /staging/twenger2/bayes_yplus-v1.3.1.sif docker-archive:/staging/twenger2/bayes_yplus-v1.3.1.tar
```

## First steps
First, we split up the data into individual data files for parallel processing. Copy the data files into `data/` first, then:
```
cd scripts/
python split_data.py
```
Individual data files are stored in `data/<datadir>/` where `<datadir>` is like `hii_noise`, `hii_intensity-noise`, `dig_noise_1.0`, `dig_intensity-noise_1.0`

## Parallel processing with SLURM
The script `scripts/fit.py` runs `bayes_yplus`'s optimization algorithm on a single pickle file. With SLURM, the data are assumed to be in `data/<datadir>/` and the results go in `results/<datadir>/` as individual pickle files.
```
python fit.py <datadir> <idx> slurm
```

The scripts `slurm/submit_<datadir>.sh` are SLURM scripts that we use to analyze each pixel in parallel. SLURM logs are written to `logs/`.
```
sbatch submit_hii_noise.sh
sbatch submit_hii_intensity-noise.sh
sbatch submit_dig_noise_1.0.sh
sbatch submit_dig_intensity-noise.sh
```

## Parallel processing with CHTC/Condor
The script `scripts/fit.py` runs `bayes_yplus`'s optimization algorithm on a single pickle file. With Condor, the data are assumed to be in the local directory as individual pickle files.
```
python fit.py <datadir> <idx> condor
```

The script `condor/run_bayes_yplus.sub` is a Condor script that we use to analyze each pixel in parallel. It handles the copying of data and results to `data/` and `results/`, respectively. Condor logs are written to `logs/`.
```
condor_submit datadir="hii_noise" run_bayes_yplus.sub
condor_submit datadir="hii_intensity-noise" run_bayes_yplus.sub
condor_submit -factory datadir="dig_noise_1.0" run_bayes_yplus.sub
condor_submit -factory datadir="dig_intensity-noise_1.0" run_bayes_yplus.sub
```