# XMFlib

**XMFlib** is a machine learning-based library for predicting pair-site probabilities, designed for surface science and materials simulation. Leveraging pre-trained neural network models, it can quickly predict various types of pair probabilities based on input interaction energy, temperature, and coverage.

---

## Features

- Supports multiple surface types (e.g., 100, 111 facets)
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

```python
from XMFlib.PairProbML import PairProbPredictor

predictor = PairProbPredictor()
result = predictor.predict(
    facet=100,                  # Facet type, options: '100' or '111'
    interaction_energy=0.2,     # Interaction energy (eV)
    temperature=400,            # Temperature (K)
    main_coverage=0.5           # Main species coverage (0~1)
)
print(result)
# Example output: {'vacancy_pair': 0.12, 'species_pair': 0.34, 'species_vacancy_pair': 0.54}
```