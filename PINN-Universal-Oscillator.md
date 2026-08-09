# Physics-Informed Neural Network (PINN): The Universal Oscillator

**Modeling Gravitational-Wave Ringdowns and Neural Action Potentials without external datasets.**

> **Core Concept:** This project demonstrates how a neural network can learn the laws of physics directly through its loss function, bypassing the need for massive training datasets. By encoding a differential equation into the PyTorch autograd graph, the AI teaches itself to simulate a damped harmonic oscillator.

---

## 1. The Interdisciplinary Physics Problem
In nature, systems that are perturbed and return to equilibrium often share the exact same mathematical grammar:
*   **Astrophysics:** Following a binary black hole merger, the remnant black hole undergoes a "ringdown," vibrating and radiating the remaining energy as gravitational waves.
*   **Neurobiology:** Following an action potential (a spike), a neuron's membrane voltage decays and oscillates back to its resting state.

Both of these phenomena can be modeled as damped harmonic oscillators, governed by the following differential equation:

$$ \frac{d^2x}{dt^2} + 2\delta \frac{dx}{dt} + \omega_0^2 x = 0 $$

Where $\delta$ represents the damping coefficient and $\omega_0$ represents the natural frequency.

## 2. The Machine Learning Approach
Traditional machine learning relies on "black box" data matching. A standard neural network would require thousands of simulated waveforms to learn this shape. 

This project uses a **Physics-Informed Neural Network (PINN)**. Instead of feeding the model data, the model is fed the laws of physics.

### Architecture & Methodology
*   **Framework:** PyTorch
*   **Network:** A simple Multi-Layer Perceptron (1 input, 32 hidden units, 1 output).
*   **Activation:** `Tanh` is used instead of `ReLU` to ensure the network is twice-differentiable (a requirement for calculating acceleration).
*   **The Loss Function:** 
    *   *Boundary Loss:* Forces the network to start at amplitude $x=1$ at time $t=0$.
    *   *Physics Loss:* Uses `torch.autograd.grad` to continuously calculate the network's first derivative (velocity) and second derivative (acceleration) with respect to time. These derivatives are plugged back into the oscillator equation. The network is penalized if the equation does not equal exactly zero.

## 3. Results
By simply asking the network to minimize the mathematical violation of the differential equation, it autonomously shaped its weights to output a perfect decaying wave.

![PINN Prediction]
<img width="1470" height="816" alt="image" src="https://github.com/user-attachments/assets/ecb432f2-4508-4b62-93e8-f7b3b2f6850d" />

*(The output of the PINN after 15,000 epochs. The model was given no external time-series data to trace; the shape is a pure mathematical consequence of the physics-informed loss function.)*

## 4. Significance
This prototype serves as a foundation for solving inverse problems in scientific machine learning. While this model acts as a forward solver (simulating the physics), the architecture can be expanded to infer hidden parameters (like a black hole's mass or a neuron's resistance) directly from sparse, noisy observational data by forcing the neural network to obey the underlying physical laws.
