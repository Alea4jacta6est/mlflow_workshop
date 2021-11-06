import argparse

from src.train_keras import train_and_save
from src.train_forest import train_and_save_tree
from src.evaluate import get_and_save_scores


EXPERIMENTS = ["Keras", "RFC"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process hyperparameters for ML pipeline")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_classes', type=int, default=10),
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--exp_name', type=str, default="Keras")
    parser.add_argument("--n_estimators", type=int, default=None)

    args = parser.parse_args()
    if args.exp_name == EXPERIMENTS[0]:
        model, model_name = train_and_save(args.num_classes,
                                            batch_size=args.batch_size,
                                            epochs=args.epochs)
    elif args.exp_name == EXPERIMENTS[1]:
        model, model_name = train_and_save_tree(args.n_estimators)
    else:
        raise ValueError("Wrong experiment name")
    scores = get_and_save_scores(model_name)
