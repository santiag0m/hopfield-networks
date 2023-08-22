from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from datasets import load_mnist_patterns


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


def plot_classic_example(n_patterns: int = 10) -> tuple[plt.Figure, plt.Axes]:
    # Load some samples from MNIST
    patterns = load_mnist_patterns(n_patterns)

    # Create a Hopfield network
    hopfield = ClassicHopfield(patterns)

    # Create a random initial state
    state = np.random.choice([-1, 1], patterns.shape[1])

    plt.ion()
    f, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].set_title("Current State")
    axs[1].set_title("Energy")
    axs[1].set_xlabel("Iterations")
    axs[1].set_box_aspect(1)
    axs[1].grid()

    state_img = None
    energy_line = None
    energy = [hopfield.energy(state)]

    # Create an animation of the network dynamics
    for _ in range(10):
        if state_img is not None:
            state_img.remove()
        if energy_line is not None:
            energy_line.remove()
        state_img = axs[0].imshow(state.reshape(28, 28), animated=True)
        state = hopfield(state)
        energy.append(hopfield.energy(state))
        (energy_line,) = axs[1].plot(energy, color="blue")

        plt.pause(1)

    return f, axs


if __name__ == "__main__":
    f, axs = plot_classic_example(n_patterns=2)
    f.savefig("classic_hopfield.png")
