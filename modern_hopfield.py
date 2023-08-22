from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

from datasets import load_mnist_patterns
from visualization import visualize_hopfield_network


class ModernHopfield:
    def __init__(self, patterns: np.ndarray):
        self.patterns = patterns  # N x F

    def __call__(self, state: np.ndarray) -> np.ndarray:
        if len(state.shape) == 1:
            state = state[np.newaxis, :]
        new_state = []
        for i in range(state.shape[0]):
            # Create an array of shape (F, F) where each row i
            # has the component state[i] flipped (i.e. 1 -> -1, -1, 1)
            state_ = state[[i], :]
            flipped_pos = np.repeat(state_, state_.shape[-1], axis=0)
            flipped_pos[np.diag_indices(state_.shape[-1])] = 1

            flipped_neg = np.repeat(state_, state_.shape[-1], axis=0)
            flipped_neg[np.diag_indices(state_.shape[-1])] = -1

            state_ = self.energy(flipped_neg) > self.energy(flipped_pos)
            state_ = state_.astype(int)
            state_[state_ == 0] = -1

            new_state.append(state_)
        return np.array(new_state)

    def energy(self, state: np.ndarray) -> float:
        lse = logsumexp(state @ self.patterns.T, axis=-1)  # S x F @ F x N = S x N
        if isinstance(lse, np.ndarray):
            if len(lse) == 1:
                lse = lse[0]
        return -1 * lse


def plot_modern_example(
    n_patterns: int = 10, output_path: Optional[str] = None
) -> tuple[plt.Figure, plt.Axes]:
    # Load some samples from MNIST
    patterns = load_mnist_patterns(n_patterns)

    # Create a Hopfield network
    hopfield = ModernHopfield(patterns)

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
    f, axs = plot_modern_example(n_patterns=2, output_path="modern_hopfield.gif")
