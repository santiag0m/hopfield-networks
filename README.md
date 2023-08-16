# Hopfield Networks

This repository contains simple implementations of the family of Hopfield Networks discussed in the paper ["Hopfield Networks is All You Need"](https://arxiv.org/abs/2008.02217), and its associated [blog post](https://ml-jku.github.io/hopfield-layers/).

## Installation

```bash
pip install -r requirements.txt
```

## Classic Hopfield Network

Memory for binary retrieval (see [Hopfield, 1982](https://authors.library.caltech.edu/7427/1/HOPpnas82.pdf))

```bash
python classic_hopfield.py
```

with output:

<img src="/classic_hopfield.png" width="512">

## Modern Hopfield Network

Improved memory for binary retrieval (see [Demircigil et al., 2017](https://arxiv.org/abs/1702.01929))

```bash
python modern_hopfield.py
```

with output:

<img src="/modern_hopfield.png" width="512">

## Modern Hopfield Network with Continuous States

The new network proposed in the paper, equivalent to the attention mechanism used in Transformers (up to linear transformations of its inputs)

```bash
python continuous_hopfield.py
```

with output:

<img src="/continuous_hopfield.png" width="512">
