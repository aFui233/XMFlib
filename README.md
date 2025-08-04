# XMFlib

[English](./README.md) | [中文](./README_CN.md)

**XMFlib** is a machine learning-based library for predicting pair-site probabilities, designed for surface science and materials simulation. Leveraging pre-trained neural network models, it can quickly predict various types of pair probabilities based on input interaction energy, temperature, and coverage.

---

## Features

- Supports multiple surface types (e.g., 100, 111 facets)
- Supports first-nearest neighbor (1NN) and second-nearest neighbor (2NN) interaction predictions
- Built-in multi-layer perceptron (MLP) models for efficient inference
- Simple and user-friendly API, easy to integrate into research and engineering projects
- Compatible with PyTorch, making it easy to extend and customize models

---

## Installation

```bash
pip install XMFlib
```

---

## Virtual Environment Setup (Recommended)

```bash
conda create --name <env_name> python=3.9
conda activate <env_name>
pip install XMFlib
```

---

## Usage Example

Basic Model Prediction (1NN interactions only)

```python
from XMFlib.PairProbML import PairProbPredictor

predictor = PairProbPredictor()
result = predictor.predict(
    facet=100,                  # Facet type, options: '100' or '111'
    interaction_energy=0.3,     # Interaction energy (eV)
    temperature=400,            # Temperature (K)
    main_coverage=0.7           # Main species coverage (0~1)
)
print("Predicted probabilities:", result)
```

**Example output:**
```
Predicted probabilities: [1.9832839222709777e-05, 0.38549050273994284, 0.6144896644208344]
```

2NN Model Prediction (considering 1NN and 2NN interactions)

```python
from XMFlib.PairProbML import PairProbPredictor

predictor = PairProbPredictor()
result_2nn = predictor.predict_2nn(
    facet=111,                     # Facet type, options: '100' or '111'
    interaction_energy_1nn=0.16,   # 1NN interaction energy (eV)
    interaction_energy_2nn=0.04,   # 2NN interaction energy (eV)
    temperature=525,               # Temperature (K)
    main_coverage=0.7              # Main species coverage (0~1)
)
print("Predicted 2NN probabilities:", result_2nn)
```

**Example output:**
```
Predicted 2NN probabilities: [0.012345678901234, 0.45678901234567, 0.53086530825309]
```

The list corresponds to:

- **Pee**: probability of a vacancy-vacancy pair (empty-empty site)
- **Paa**: probability of a specie-specie pair (specie-specie)
- **Pae**: probability of a specie-vacancy pair (specie-empty site)