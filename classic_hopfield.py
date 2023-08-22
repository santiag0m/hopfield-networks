from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from datasets import load_mnist_patterns
from visualization import visualize_hopfield_network


class ClassicHopfield:
    def __init__(self, patterns: np.ndarray, threshholds: Optional[np.ndarray] = None):
        if threshholds is None:
            threshholds = np.zeros(patterns.shape[1])

        self.patterns = patterns
        self.threshholds = threshholds
        self.weights = self._calculate_weights()

    def _calculate_weights(self) -> np.ndarray:
        weights = self.patterns.T @ self.patterns  # (F x N) @ (N x F) = F x F
        np.fill_diagonal(weights, 0)
        return weights

    def __call__(self, state: np.ndarray) -> np.ndarray:
        return np.sign(state @ self.weights - self.threshholds)

    def energy(self, state: np.ndarray) -> float:
        return -0.5 * state @ self.weights @ state + state @ self.threshholds


def plot_classic_example(
    n_patterns: int = 10, output_path: Optional[str] = None
) -> tuple[plt.Figure, plt.Axes]:
    # Load some samples from MNIST
    patterns = load_mnist_patterns(n_patterns)

    # Create a Hopfield network
    hopfield = ClassicHopfield(patterns)

    # Create a random initial state
    state = np.random.choice([-1, 1], patterns.shape[1])

    plt.ion()
    f, axs = visualize_hopfield_network(
        hopfield_network=hopfield,
        energy_function=hopfield.energy,
        initial_state=state,
        output_path=output_path,
    )
    return f, axs


if __name__ == "__main__":
    f, axs = plot_classic_example(n_patterns=2, output_path="classic_hopfield.gif")
