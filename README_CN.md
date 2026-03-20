# XMFlib

[English](./README.md) | [中文](./README_CN.md)

**XMFlib** 是一个面向表面科学的轻量级机器学习推理库。

当前聚焦三个核心主题：

1. **PairProbML**：位点对概率预测
2. **CovML**：覆盖度预测
3. **快速集成**：预训练模型 + 简洁 API

---

## 安装

```bash
pip install XMFlib
```

---

## 虚拟环境配置（建议）

```bash
conda create --name <env_name> python=3.9
conda activate <env_name>
pip install XMFlib
```

---

## 1. PairProbML

- 支持 `100` / `111` 晶面配对概率预测
- 支持 1NN 与 2NN 推理
- 输出顺序为 `[Pee, Paa, Pae]`

1NN 示例：

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

2NN 示例：

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

- 根据相互作用能、吸附能和温度预测覆盖度
- 输出顺序为 `[A_Coverage, E_Coverage]`

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