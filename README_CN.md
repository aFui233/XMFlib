# XMFlib

[English](./README.md) | [中文](./README_CN.md)

**XMFlib** 是一个基于机器学习的位点对概率预测库，适用于表面科学和材料模拟领域。它通过预训练的神经网络模型，能够根据输入的相互作用能、温度和覆盖度，快速预测不同类型的配对概率。

---

## 特性

- 支持多种表面类型（如 100, 111 晶面）
- 支持第一近邻(1NN)和第二近邻(2NN)相互作用考虑
- 内置多层感知机(MLP)模型，推理高效
- 简单易用的 API，便于集成到科研和工程项目
- 兼容 PyTorch，易于扩展和自定义模型

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

___

## 使用示例

基本模型预测（仅考虑第一近邻相互作用）

```python
from XMFlib.PairProbML import PairProbPredictor

predictor = PairProbPredictor()
result = predictor.predict(
    facet=100,                  # 晶面类型，可选 '100' 或 '111'
    interaction_energy=0.3,     # 相互作用能 (eV)
    temperature=400,            # 温度 (K)
    main_coverage=0.7           # 主要物种覆盖度 (0~1)
)
print("Predicted probabilities:", result)
```

**输出示例：**
```
Predicted probabilities: [0.002484329044818878, 0.38522598147392273, 0.5955939292907715]
```

2NN模型预测（同时考虑第一近邻和第二近邻相互作用）

```python
from XMFlib.PairProbML import PairProbPredictor

predictor = PairProbPredictor()
result_2nn = predictor.predict_2nn(
    facet=111,                     # 晶面类型，可选 '100' 或 '111'
    interaction_energy_1nn=0.16,   # 第一近邻相互作用能 (eV)
    interaction_energy_2nn=0.04,   # 第二近邻相互作用能 (eV)
    temperature=525,               # 温度 (K)
    main_coverage=0.7              # 主要物种覆盖度 (0~1)
)
print("Predicted 2NN probabilities:", result_2nn)
```

**输出示例：**
```
Predicted 2NN probabilities: [0.012345678901234, 0.45678901234567, 0.53086530825309]
```

该列表中的各元素含义如下：

- **Pee**：空位-空位对的概率（即两个相邻空位点同时出现的概率）
- **Paa**：主物种-主物种对的概率（即两个相邻主物种吸附位点同时出现的概率）
- **Pae**：主物种-空位对的概率（即一个主物种和一个空位相邻出现的概率）