import numpy as np
from sklearn.datasets import fetch_openml


def load_mnist_patterns(n: int = 10, binary: bool = True) -> np.ndarray:
    print("Loading MNIST...")
    mnist = fetch_openml(
        "mnist_784", version=1, cache=True, as_frame=False, parser="liac-arff"
    )
    idx = np.random.randint(0, 70000, n)
    patterns = mnist["data"][idx]
    if binary:
        patterns[patterns < 128] = -1
        patterns[patterns >= 128] = 1
    else:
        patterns = patterns / 255.0
    return patterns
