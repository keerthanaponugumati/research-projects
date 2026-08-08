# Toy Gravitational-Wave Parameter Recovery Prototype

**Simulation-based data generation + 1D CNN mass regression + path toward Neural Posterior Estimation**

- **Prepared for:** Professors, research mentors, and technical reviewers
- **Prepared by:** KEERTHANA PONUGUMATI
- **Date:** [Insert Date]
- **Repository/Notebook:** [Insert link, if available]

> **Honest scope statement:** This project is a toy, end-to-end prototype. It does not use real LIGO detector pipelines, physically complete waveform models, colored detector noise, or Bayesian sampling. Its value is that it reproduces the conceptual skeleton of gravitational-wave parameter recovery: simulate signals from parameters, add noise, train a model, and evaluate whether source parameters can be recovered.

### Overview of Demonstration
| What the project does | What it demonstrates |
| :--- | :--- |
| Generates chirp-like signals from two input masses. | Understanding of the forward simulator idea: parameters ➔ waveform. |
| Adds Gaussian noise to create detector-like inputs. | Understanding that inference is performed on data, not clean signals. |
| Trains a PyTorch 1D CNN to predict component masses. | Ability to connect time-series ML with physics-inspired features. |
| Evaluates predictions with loss curves and true-vs-predicted plots. | Awareness that model performance must be validated outside training data. |
| Frames the next step as posterior inference, not only point prediction. | Awareness of uncertainty quantification and Neural Posterior Estimation. |

---

## Abstract
This project implements a small end-to-end prototype for gravitational-wave-inspired parameter recovery. The system samples two black hole component masses, generates a chirp-like toy waveform, adds random Gaussian noise, and trains a one-dimensional convolutional neural network to estimate the masses from the noisy time-series data. The prototype is intentionally simplified, but it follows the same broad logic used in simulation-based inference: define source parameters, simulate observations, learn an inverse map from data back to parameters, and evaluate recovery quality on held-out examples. The current model produces point estimates for `m1` and `m2`; the planned research direction is to replace point regression with a neural posterior estimator that returns a distribution `p(theta | d)`, enabling uncertainty-aware inference.

## 1. Motivation and scientific context
Gravitational waves from compact binary inspirals are produced by systems such as binary black holes, binary neutron stars, or neutron-star-black-hole binaries. As the objects orbit closer together, the emitted signal changes over time. In the detector band, this can appear as a short signal whose frequency and amplitude increase, commonly described as a chirp. Component masses influence the signal duration and frequency evolution, which makes mass recovery a natural toy problem for learning the inverse relationship between waveform shape and source parameters.

In real gravitational-wave astronomy, parameter estimation is usually framed statistically: given detector data `d` and model parameters `theta`, infer a posterior distribution `p(theta | d)`. This posterior captures both the best-supported parameter values and the uncertainty caused by detector noise, parameter degeneracies, and modeling assumptions. This toy project starts with the simpler supervised-learning version of the problem, then identifies a path toward posterior estimation.

## 2. Research question
- **Core question:** Can a neural network learn to recover two source masses from noisy, chirp-like simulated time-series signals?
- **Secondary question:** How can this point-estimation prototype be upgraded into a simulation-based inference system that reports uncertainty, not just best guesses?

The implemented pipeline can be summarized as:
`sample masses` ➔ `simulate chirp waveform` ➔ `add Gaussian noise` ➔ `train 1D CNN` ➔ `predict masses` ➔ `evaluate on held-out test signals`

## 4. System components

| Component | What was implemented | Research-grade upgrade path |
| :--- | :--- | :--- |
| **Waveform simulator** | Chirp-like 1D time series generated from `m1` and `m2`. | Use physical compact-binary waveform approximants (e.g., via established GW software ecosystems). |
| **Noise model** | Additive Gaussian noise: `d(t) = h(t; theta) + n(t)`. | Use detector power spectral densities, whitening, real noise segments, and nonstationary noise conditioning. |
| **Dataset generator** | Synthetic pairs of noisy signals and true component masses. | Define astrophysically meaningful priors, SNR ranges, train/validation/test splits, and out-of-distribution tests. |
| **Neural network** | 1D CNN trained to predict two masses from the noisy time series. | Move from point regression to posterior density estimation using normalizing flows or related NPE methods. |
| **Evaluation** | Training loss and test-set true-vs-predicted plots. | Report MAE/RMSE, calibration, posterior coverage, residual trends, and comparison with Bayesian baselines. |

## 5. Toy waveform simulator
- **Inputs:** The simulator takes two component masses, `m1` and `m2`. In a physically complete compact-binary model, the parameter vector would include additional quantities (spins, luminosity distance, sky location, etc.). For this prototype, `theta = (m1, m2)`.
- **Output:** The output is a one-dimensional chirp-like time series `h(t; theta)`. The toy waveform is designed to have the qualitative behavior expected from an inspiral-like signal: frequency increases with time, and amplitude can also grow toward the end of the signal. The implementation is not a numerical-relativity or post-Newtonian waveform model; it is a simple controlled simulator for machine-learning experiments.

```python
theta = (m1, m2)
h_clean = toy_chirp(t, m1, m2)
