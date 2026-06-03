# 复现报告

本报告记录本仓库 Python 3 重写版的实验结果。原始论文数值来自 *Stealing Machine Learning Models via Prediction APIs*，USENIX Security 2016。

---

## Phase 1：二元 Logistic Regression 方程求解

对应论文：§4.1 Binary logistic regression。

实验入口：

```bash
python -m src.equation_solving.run_binary_lr
```

实验设置：

- 数据集：`sklearn.datasets.load_breast_cancer`
- 特征缩放：`[-1, 1]`
- 受害者模型：`sklearn.linear_model.LogisticRegression`
- 攻击能力：只能调用 `query_proba(x)` / `query_label(x)`，不读取模型参数
- 攻击方式：查询 `d + 1` 个线性无关点，解 `[X, 1] theta = logit(p)`
- 均匀评估点：10,000

当前结果（Python 3.14.5，seed=0）：

| 指标 | 数值 |
|---|---:|
| 特征维度 `d` | 30 |
| 类别数 | 2 |
| 提取查询数 | 31 |
| 增广矩阵秩 | 31 |
| 线性系统残差范数 | `3.625e-14` |
| 测试集一致率 | 100.0000% |
| 均匀采样一致率 | 100.0000% |
| 测试集平均 TV 距离 | `4.564e-15` |
| 均匀采样平均 TV 距离 | `1.090e-15` |

结论：复现了论文最核心的二元 LR 方程求解攻击。30 维模型只需 `d + 1 = 31` 次 confidence query，即可在测试集和均匀采样输入空间上达到 100% 功能一致，概率输出误差接近数值精度。

备注：当前系统 Python 3.14 环境会打印一个用户级 `distutils-precedence.pth` 警告：

```text
ModuleNotFoundError: No module named '_distutils_hack'
```

该警告来自本机用户 site-packages 配置，不影响实验运行和结果。

---

## Phase 2：多类 Logistic Regression 方程求解

对应论文：§4.1 Multiclass LRs and Multilayer Perceptrons。

实验入口：

```bash
python -m src.equation_solving.run_multiclass_lr
```

也可以单独运行：

```bash
python -m src.equation_solving.run_multiclass_lr --dataset iris --model softmax
python -m src.equation_solving.run_multiclass_lr --dataset iris --model ovr
python -m src.equation_solving.run_multiclass_lr --dataset digits --model softmax
python -m src.equation_solving.run_multiclass_lr --dataset digits --model ovr
```

实验设置：

- 数据集：`sklearn.datasets.load_iris`、`sklearn.datasets.load_digits`
- 特征缩放：`[-1, 1]`
- 受害者模型：
  - softmax：`sklearn.linear_model.LogisticRegression`
  - OvR：`sklearn.multiclass.OneVsRestClassifier(LogisticRegression)`
- 攻击能力：只能调用 `query_proba(x)` / `query_label(x)`，不读取目标模型参数
- 攻击方式：
  - softmax：解 `log(p_i / p_ref)` 的线性系统作为闭式初始化，再用 soft-label cross-entropy 验证/优化
  - OvR：用 oracle 概率作为 soft labels，最小化归一化 OvR sigmoid 的 cross-entropy
- 默认预算：`1.0 * c * (d + 1)`，即约 1 次查询 / 参数
- 均匀评估点：10,000

当前结果（Python 3.14.5，seed=0，budget_multiplier=1.0）：

| 数据集 | 模型 | d | c | 参数量 | 提取查询数 | 测试集一致率 | 均匀一致率 | 测试 TV | 均匀 TV |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Iris | softmax | 4 | 3 | 15 | 15 | 100.0000% | 100.0000% | `1.091e-16` | `2.934e-16` |
| Iris | OvR | 4 | 3 | 15 | 15 | 100.0000% | 100.0000% | `6.755e-07` | `7.619e-07` |
| Digits | softmax | 64 | 10 | 650 | 650 | 100.0000% | 100.0000% | `3.794e-16` | `1.710e-15` |
| Digits | OvR | 64 | 10 | 650 | 650 | 100.0000% | 100.0000% | `3.946e-05` | `7.366e-06` |

结论：复现了论文中“多类 LR 约 1 query / parameter 即可提取”的核心趋势。Digits 的 softmax 模型为 64 维、10 类、650 个参数，用 650 次 confidence query 即可在测试集和 10,000 个均匀采样点上达到 100% 一致；这也对应论文在线 Amazon Digits 案例中 650 次查询的量级。

实现备注：

- 现代 `scikit-learn` 已移除 `LogisticRegression(multi_class=...)` 参数。本仓库用新版默认 multinomial LR 表示 softmax，用 `OneVsRestClassifier` 显式构造 OvR。
- softmax 参数有平移不唯一性，因此报告以功能一致率和概率 TV 距离为主，不比较参数逐项误差。
