# XMFlib

**XMFlib** 是一个基于机器学习的配对概率预测库，适用于表面科学和材料模拟领域。它通过预训练的神经网络模型，能够根据输入的相互作用能、温度和覆盖度，快速预测不同类型的配对概率。

---

## 特性

- 支持多种表面类型（如 100, 111 晶面）
- 内置多层感知机（MLP）模型，推理高效
- 简单易用的 API，便于集成到科研和工程项目
- 兼容 PyTorch，易于扩展和自定义模型

---

## 安装

```bash
pip install XMFlib
```

---

## 使用示例

```python
from PairProbML import PairProbPredictor

predictor = PairProbPredictor()
result = predictor.predict(
    facet=100,                  # 晶面类型，可选 '100' 或 '111'
    interaction_energy=0.2,    # 相互作用能 (eV)
    temperature=400,            # 温度 (K)
    main_coverage=0.5           # 主组分覆盖度 (0~1)
)
print(result)
# 输出示例: {'vacancy_pair': 0.12, 'species_pair': 0.34, 'species_vacancy_pair': 0.54}
```

