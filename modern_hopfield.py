import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt

from datasets import load_mnist_patterns


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


def plot_modern_example(n_patterns: int = 10) -> tuple[plt.Figure, plt.Axes]:
    # Load some samples from MNIST
    patterns = load_mnist_patterns(n_patterns)

    # Create a Hopfield network
    hopfield = ModernHopfield(patterns)

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
    f, axs = plot_modern_example(n_patterns=2)
    f.savefig("modern_hopfield.png")
