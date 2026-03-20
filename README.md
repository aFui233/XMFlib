# XMFlib

[English](./README.md) | [中文](./README_CN.md)

**XMFlib** is a machine learning-based library for surface science and materials simulation. It currently provides two modules: **PairProbML** for pair-site probability prediction and **CovML** for coverage prediction from interaction energy, adsorption energy, and temperature.

---

## Features

- Supports multiple surface types (e.g., 100, 111 facets)
- Supports first-nearest neighbor (1NN) and second-nearest neighbor (2NN) interaction predictions
- Supports coverage prediction based on interaction energy and adsorption energy
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

Coverage Prediction (CovML)

```python
from XMFlib.CovML import CovPredictor

predictor = CovPredictor()
result = predictor.predict(
    facet=100,                  # Facet type, options: '100' or '111'
    interaction_energy=0.18,     # Interaction energy (eV)
    adsorption_energy=-0.86,      # Adsorption energy (eV)
    temperature=400             # Temperature (K)
)
print("Predicted coverage:", result)
```

**Example output:**
```
Predicted coverage: [0.6738327667075609, 0.32616723329243913]
```

Result note: the first value is the species coverage (A_Coverage), and the second value is the vacancy coverage (E_Coverage).