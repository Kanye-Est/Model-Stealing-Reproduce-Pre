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

