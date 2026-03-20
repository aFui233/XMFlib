# XMFlib

[English](./README.md) | [中文](./README_CN.md)

**XMFlib** is a lightweight ML inference library for surface science.

It focuses on three core topics:

1. **PairProbML**: pair-site probability prediction
2. **CovML**: coverage prediction
3. **Quick integration**: simple API and pretrained models

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

## 1. PairProbML

- Predict pair probabilities on `100` / `111` facets
- Supports 1NN and 2NN inference
- Output order: `[Pee, Paa, Pae]`

Basic 1NN example:

```python
from XMFlib.PairProbML import PairProbPredictor

predictor = PairProbPredictor()
result = predictor.predict(
    facet=100,
    interaction_energy=0.3,
    temperature=400,
    main_coverage=0.7
)
print("Predicted probabilities:", result)
```

2NN example:

```python
from XMFlib.PairProbML import PairProbPredictor

predictor = PairProbPredictor()
result_2nn = predictor.predict_2nn(
    facet=111,
    interaction_energy_1nn=0.16,
    interaction_energy_2nn=0.04,
    temperature=525,
    main_coverage=0.7
)
print("Predicted 2NN probabilities:", result_2nn)
```

---

## 2. CovML

- Predict coverage from interaction energy, adsorption energy, and temperature
- Output order: `[A_Coverage, E_Coverage]`

```python
from XMFlib.CovML import CovPredictor

predictor = CovPredictor()
result = predictor.predict(
    facet=100,
    interaction_energy=0.18,
    adsorption_energy=-0.86,
    temperature=400
)
print("Predicted coverage:", result)
```