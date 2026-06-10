import argparse
import csv
import json
from pathlib import Path

import numpy as np

from mlp import MLP


def parse_hidden_layers(value):
    layers = [int(item.strip()) for item in value.split(",") if item.strip()]

    if not layers:
        raise argparse.ArgumentTypeError("Informe ao menos uma camada oculta.")

    if any(layer <= 0 for layer in layers):
        raise argparse.ArgumentTypeError("Todas as camadas ocultas precisam ser positivas.")

    return layers


def load_mnist(train_limit=None, test_limit=None):
    mnist = None

    try:
        from tensorflow.keras.datasets import mnist as tensorflow_mnist

        mnist = tensorflow_mnist
    except ModuleNotFoundError:
        try:
            from keras.datasets import mnist as keras_mnist

            mnist = keras_mnist
        except ModuleNotFoundError:
            pass

    if mnist is not None:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        return prepare_mnist_arrays(x_train, y_train, x_test, y_test, train_limit, test_limit)

    try:
        from torchvision.datasets import MNIST
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Instale tensorflow, keras ou torchvision para carregar o MNIST."
        ) from error

    train_dataset = MNIST(root="data", train=True, download=True)
    test_dataset = MNIST(root="data", train=False, download=True)
    x_train = train_dataset.data.numpy()
    y_train = train_dataset.targets.numpy()
    x_test = test_dataset.data.numpy()
    y_test = test_dataset.targets.numpy()

    return prepare_mnist_arrays(x_train, y_train, x_test, y_test, train_limit, test_limit)


def prepare_mnist_arrays(x_train, y_train, x_test, y_test, train_limit=None, test_limit=None):
    x_train = x_train.reshape(x_train.shape[0], -1).astype(float) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1).astype(float) / 255.0

    if train_limit is not None:
        x_train = x_train[:train_limit]
        y_train = y_train[:train_limit]

    if test_limit is not None:
        x_test = x_test[:test_limit]
        y_test = y_test[:test_limit]

    return x_train, y_train, x_test, y_test


def save_history(history, results_dir):
    results_dir.mkdir(parents=True, exist_ok=True)

    with (results_dir / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    metric_names = list(history.keys())
    epochs = range(1, len(history["loss"]) + 1)

    with (results_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", *metric_names])

        for index, epoch in enumerate(epochs):
            writer.writerow([epoch, *[history[name][index] for name in metric_names]])


def save_summary(summary, results_dir):
    with (results_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def save_plots(history, results_dir):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(history["loss"], label="train")

    if "val_loss" in history:
        plt.plot(history["val_loss"], label="test")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "loss.png")
    plt.close()

    plt.figure()
    plt.plot(history["accuracy"], label="train")

    if "val_accuracy" in history:
        plt.plot(history["val_accuracy"], label="test")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "accuracy.png")
    plt.close()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-layers", type=parse_hidden_layers, default=[256, 128])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    return parser


def run_experiment(
    hidden_layers,
    epochs,
    batch_size,
    learning_rate,
    seed,
    results_dir,
    train_limit=None,
    test_limit=None,
):
    if epochs <= 0:
        raise ValueError("epochs precisa ser positivo.")

    if batch_size <= 0:
        raise ValueError("batch-size precisa ser positivo.")

    results_dir = Path(results_dir)
    x_train, y_train, x_test, y_test = load_mnist(train_limit, test_limit)
    architecture = [x_train.shape[1], *hidden_layers, 10]
    model = MLP(architecture, learning_rate=learning_rate, random_seed=seed)

    history = model.train(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
    )

    test_loss = float(model.loss(x_test, y_test))
    test_accuracy = float(model.accuracy(x_test, y_test))
    summary = {
        "architecture": architecture,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }

    save_history(history, results_dir)
    save_summary(summary, results_dir)
    save_plots(history, results_dir)

    print(f"Arquitetura: {architecture}")
    print(f"Loss teste: {test_loss:.4f}")
    print(f"Acuracia teste: {test_accuracy:.4f}")
    print(f"Resultados salvos em: {results_dir}")

    return summary


def main():
    args = build_parser().parse_args()
    run_experiment(
        hidden_layers=args.hidden_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        results_dir=args.results_dir,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
    )


if __name__ == "__main__":
    main()
