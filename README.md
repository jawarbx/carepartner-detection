# carepartner-detection

## Installation
1. Install the packages in `requirements.txt` in your preferred virtual environment

2. Create a `.env` using `env.example`. All environment variables in env.example must be filled out.

## Usage
To run a job locally, run the following command:
```bash
$ source .env
$ ./shap_analysis.sh
```

To run a batch job on slurm, run the following command:
```bash
$ source .env
$ sbatch --parsable \
	--job-name=${EXPERIMENT_NAME} \
	--output=${OUTPUT_DIR}/${EXPERIMENT_NAME}_%j.out \
	--error=${OUTPUT_DIR}/${EXPERIMENT_NAME}_%j.err \
	shap_analysis.sh
```
