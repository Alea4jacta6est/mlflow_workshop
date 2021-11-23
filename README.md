# MLFlow tutorial - MNIST training 

## Before making experiments

1. Clone this repository

2. Activate venv for Python 3.9 and `pip install -r requirements.txt` 

3. Install tensorflow for your OS


## How to run MLflow locally?

Run `mlflow ui`

Make experiments using `python main.py (--n_estimators 50 --exp_name "RFC" / "Keras")` combining different args

Open http://0.0.0.0:5000 to check experiments

*If you want to set up remote workflow, this doc can help: https://www.mlflow.org/docs/latest/tracking.html#scenario-4-mlflow-with-remote-tracking-server-backend-and-artifact-stores

Materials:
https://www.mlflow.org/
