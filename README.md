## SIGMA: Self-guided Integrated Gradient Method for Attribution
<p align="center"><em>A path-based attribution method guided by stochastic perturbations</em></p>

---

## Updates
- **09/06/2025** – Initial SIGMA code released with example notebook and test images.

---

## Overview

SIGMA (Self-guided Integrated Gradient Method for Attribution) is a path-based explainable AI (XAI) method designed to identify which input features most influence a model's prediction.

Unlike traditional Integrated Gradients [1], SIGMA uses the Simultaneous Perturbation Stochastic Approximation (SPSA) algorithm [2] to dynamically guide the integration path. This avoids reliance on a fixed baseline and enables more informative, input-specific paths.

---

## Usage

Our method is implemented in [`SIGMA.py`](SIGMA.py), and a full demo is available in the [`SIGMA_example.ipynb`](SIGMA_example.ipynb) notebook.

To use SIGMA in your own project:

```python
from SIGMA import SIGMA_attribution

# Example usage for a given model and image

attribution = SIGMA_attribution(model, image, target_class, n, beta, alpha, epsilon, model_preprocess_fn)

# Arguments:
# model: A TensorFlow model (e.g. InceptionV3, ResNet50, etc.)
# image: Input image as a tensor
# target_class: Integer class index to explain (from model prediction)
# n: Number of paths to average (e.g. 5–10)
# beta: Perturbation magnitude (SPSA step size) (e.g. 0.01–0.5)
# alpha: Step size (e.g. 0.1–1)
# epsilon: Confidence threshold for stopping path integration (recommend 0.01-0.05)
# model_preprocess_fn: A preprocessing function matching the model (e.g., keras.applications.inception_v3.preprocess_input)

```

---
## Acknowledgements

The following libraries and resources were used in benchmarking SIGMA against existing attribution methods:

- [**Saliency** by PAIR](https://github.com/PAIR-code/saliency): This library was used to run and compare SIGMA against other saliency-based explainability methods such as Integrated Gradients and Guided Integrated Gradients.

- [**XAI-BENCH**](https://github.com/XAIdataset/XAIdataset.github.io): We used XAI-BENCH benchmark datasets and quantitative alignment metrics to evaluate SIGMA’s attribution performance against established XAI methods.
---

## References

1. Sundararajan, M., Taly, A., & Yan, Q. (2017). *Axiomatic Attribution for Deep Networks*. In Proceedings of the 34th International Conference on Machine Learning (ICML). [arXiv:1703.01365](https://arxiv.org/abs/1703.01365)

2. Spall, J.C. (1992). *Multivariate Stochastic Approximation Using a Simultaneous Perturbation Gradient Approximation*. IEEE Transactions on Automatic Control, 37(3), 332–341. [DOI:10.1109/9.119632](https://doi.org/10.1109/9.119632)

---


