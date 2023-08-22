from typing import Callable, Optional

import matplotlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def visualize_hopfield_network(
    hopfield_network: Callable[[np.ndarray], np.ndarray],
    energy_function: Callable[[np.ndarray], float],
    initial_state: np.ndarray,
    output_path: Optional[str] = None,
    steps: int = 10,
):
    state = initial_state

    if output_path is not None:
        matplotlib.use("Agg")

    f, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].set_title("Initial State")
    axs[1].set_title("Current State")
    axs[2].set_title("Energy")
    axs[2].set_xlabel("Iterations")
    axs[2].set_box_aspect(1)
    axs[2].grid()
    axs[2].yaxis.set_major_formatter(FormatStrFormatter("%.2E"))

    axs[0].imshow(state.reshape(28, 28))

    state_img = None
    energy_line = None
    energy = [energy_function(state)]

    frames = []
    # Create an animation of the network dynamics
    for _ in range(steps):
        if state_img is not None:
            state_img.remove()
        if energy_line is not None:
            energy_line.remove()
        state_img = axs[1].imshow(state.reshape(28, 28), animated=True)
        state = hopfield_network(state)
        energy.append(energy_function(state))
        (energy_line,) = axs[2].plot(energy, color="blue")

        if output_path is not None:
            f.canvas.draw_idle()
            frames.append(_figure_to_frame(f))
        else:
            plt.pause(1)

    if output_path is not None:
        # Convert the list of figures to a GIF
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=500,  # Duration for each frame
            loop=0,
        )  # Number of times the GIF should loop (0 means infinite)

    return f, axs


def _figure_to_frame(fig: plt.Figure) -> Image:
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return Image.fromarray(image)
