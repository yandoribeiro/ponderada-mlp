from pathlib import Path

from train import run_experiment


HIDDEN_LAYERS = [128, 64]
EPOCHS = 120
BATCH_SIZE = 64
LEARNING_RATE = 0.01
SEED = 42
RESULTS_DIR = Path("results/experimento_2")


def main():
    run_experiment(
        hidden_layers=HIDDEN_LAYERS,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        seed=SEED,
        results_dir=RESULTS_DIR,
    )


if __name__ == "__main__":
    main()
