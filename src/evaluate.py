import os
import json

from keras.models import load_model
from sklearn.metrics import accuracy_score
import joblib

from src.preprocess import get_processed_data


def add_metrics_to_json(save_filename: str, updated_scores: str):
    """Appends scores from new experiments

    Args:
        save_filename (str): path to scores.json
        updated_scores (dict): dict with scores
    """
    os.makedirs("reports", exist_ok=True)
    if save_filename.split("/")[1] in os.listdir("reports"):
        with open(save_filename, "r") as file:
            scores = json.load(file)
        updated_scores = {**scores, **updated_scores}
    with open(save_filename, "w") as file:
        json.dump(updated_scores, file, indent=4, ensure_ascii=False)


def get_and_save_scores(model_path: str, save_filename: str = "reports/scores.json"):
    """Gets cached test data, loads model and counts scores

    Args:
        model_name (str): name of the model
        save_filename (str): a path to save the file
    """
    _, _, x_test, y_test = get_processed_data()
    if "h5" in model_path:
        model = load_model(model_path)
        score = model.evaluate(x_test, y_test, verbose=0)
        final_scores = {'Test accuracy': round(score[1], 4)}
        print(final_scores)
    else:
        nsamples, nx, ny, _ = x_test.shape
        x_test = x_test.reshape((nsamples, nx * ny))
        model = joblib.load(model_path)
        forest_output = model.predict(x_test)
        print(accuracy_score(y_test, forest_output))
        final_scores = {
            "Test accuracy":
            round(accuracy_score(y_test, forest_output), 4)
        }

    if save_filename:
        add_metrics_to_json(save_filename, final_scores)
    return final_scores


if __name__ == "__main__":
    get_and_save_scores()
