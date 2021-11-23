import argparse
import os
import mlflow
import mlflow.sklearn

from src.train_keras import train_and_save
from src.train_forest import train_and_save_tree
from src.evaluate import get_and_save_scores

MLFLOW_TRACKING_URI = "http://0.0.0.0:5000"
# For example, you need to store data in GCP:
os.environ[
    'GOOGLE_APPLICATION_CREDENTIALS'] = 'secrets/mlflow-server-credentials.json'
EXPERIMENTS = ["Keras", "RFC"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process hyperparameters for ML pipeline")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_classes', type=int, default=10),
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--exp_name', type=str, default="Keras")
    parser.add_argument("--n_estimators", type=int, default=None)

    args = parser.parse_args()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(args.exp_name)
    with mlflow.start_run():
        if args.exp_name == EXPERIMENTS[0]:
            model, model_name = train_and_save(args.num_classes,
                                               batch_size=args.batch_size,
                                               epochs=args.epochs)
            mlflow.log_param("batch_size", args.batch_size)
            mlflow.log_param("num_classes", args.num_classes)
            mlflow.log_param("epochs", args.epochs)
            mlflow.keras.log_model(model, model_name)
        elif args.exp_name == EXPERIMENTS[1]:
            model, model_name = train_and_save_tree(args.n_estimators)
            mlflow.log_param("n_estimators", args.n_estimators)
            mlflow.sklearn.log_model(model, model_name)
        else:
            raise ValueError("Wrong experiment name")
        scores = get_and_save_scores(model_name)
        mlflow.log_metrics(scores)
