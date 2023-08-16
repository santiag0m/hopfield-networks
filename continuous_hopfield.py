import numpy as np
from scipy.special import logsumexp, softmax
import matplotlib.pyplot as plt

from datasets import load_mnist_patterns


class ContinuousHopfield:
    def __init__(self, patterns: np.ndarray, beta: float = 1.0):
        self.patterns = patterns  # N x F

        self.beta = beta
        self.n = self.patterns.shape[0]
        self.m = (self.patterns**2).sum(axis=-1).max()

    def __call__(self, state: np.ndarray) -> np.ndarray:
        if len(state.shape) == 1:
            state = state[np.newaxis, :]

        state = softmax(self.beta * state @ self.patterns.T, axis=-1) @ self.patterns
        return state

    def energy(self, state: np.ndarray) -> float:
        lse = (
            logsumexp(self.beta * (state @ self.patterns.T), axis=-1) / self.beta
        )  # S x F @ F x N = S x N

        state_norm = (state**2).sum(axis=-1) * 0.5
        n = np.log(self.n) / self.beta
        m = self.m**2 * 0.5

        energy = -1 * lse + state_norm + n + m

        if isinstance(energy, np.ndarray):
            if len(energy) == 1:
                energy = energy[0]

        return energy


def plot_continuous_example(n_patterns: int = 10) -> tuple[plt.Figure, plt.Axes]:
    # Load some samples from MNIST
    patterns = load_mnist_patterns(n_patterns, binary=False)

    # Create a Hopfield network
    hopfield = ContinuousHopfield(patterns)

    # Create a random initial state
    state = np.random.randn(patterns.shape[1])

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

    return f, axs


if __name__ == "__main__":
    f, axs = plot_continuous_example(n_patterns=2)
    f.savefig("continuous_hopfield.png")
