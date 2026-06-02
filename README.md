# Model-Stealing-Reproduce-Pre

复现并讲解经典论文 **《Stealing Machine Learning Models via Prediction APIs》**（Tramèr et al., USENIX Security 2016）——模型窃取（Model Extraction）攻击的开山之作。

本仓库用于：① 组会论文分享（pre）；② 用现代 **Python 3** 复现论文核心攻击。

> 📄 原论文代码（Python 2，已停维护）：https://github.com/ftramer/Steal-ML
> 本仓库**不是 fork**，而是把核心算法用干净的 Python 3 + 现代 numpy/sklearn 重写，便于阅读、运行与教学。

---

## 复现的四个攻击

| # | 攻击 | 论文章节 | 是否需要联网/账号 | 目录 |
|---|---|---|---|---|
| 1 | **方程求解** Equation-Solving（LR / softmax / OvR / MLP） | §4 | 否（纯本地） | `src/equation_solving/` |
| 2 | **决策树路径查找** Path-Finding | §4.2 | 否（本地训练树当黑盒） | `src/decision_tree/` |
| 3 | **仅标签** Label-Only（Lowd-Meek + retraining） | §5 | 否 | `src/label_only/` |
| 4 | **KLR 训练数据泄露**可视化 | §4.1.3 | 否 | `src/klr_leakage/` |

所有攻击都在**本地训练一个"受害者模型"当作黑盒 oracle**（只暴露查询接口、统计查询次数），从而无需 AWS/BigML 账号即可复现论文的核心现象。

---

## 目录结构

```
Model-Stealing-Reproduce-Pre/
├── README.md
├── requirements.txt
├── docs/
│   ├── paper-notes.md            # 论文精读笔记（中文）
│   ├── presentation-outline.md   # 组会 pre 大纲
│   └── reproduction-report.md    # 复现结果 vs 论文数值
├── src/
│   ├── common/                   # 数据集加载 + 黑盒 oracle 抽象 + 误差度量
│   ├── equation_solving/         # 攻击一
│   ├── decision_tree/            # 攻击二
│   ├── label_only/               # 攻击三
│   └── klr_leakage/              # 攻击四
├── experiments/results/          # 实验输出（csv）
└── figures/                      # 生成的图
```

---

## 环境

```bash
# 推荐 Python 3.10+（开发时用 conda base 的 3.12；3.14 亦可）
pip install -r requirements.txt
```

依赖：`numpy / scipy / scikit-learn / pandas / matplotlib`。

---

## 快速开始

```bash
cd Model-Stealing-Reproduce-Pre

# 攻击一：方程求解（在多个数据集上，画"查询数 vs 一致率"）
python -m src.equation_solving.run_equation_solving

# 攻击二：决策树路径查找（本地树当黑盒，验证完美还原）
python -m src.decision_tree.run_tree_extraction

# 攻击三：Label-Only（三种 retraining + Lowd-Meek 对比曲线）
python -m src.label_only.run_label_only

# 攻击四：KLR 训练数据泄露（并排展示 原图 vs 提取图）
python -m src.klr_leakage.run_klr_leakage
```

图输出到 `figures/`，数值输出到 `experiments/results/`。

---

## 复现结果

跑完后见 [`docs/reproduction-report.md`](docs/reproduction-report.md)（含与论文数值的逐项对比）。

---

## 与原始 Steal-ML 的主要差异

| | 原始 Steal-ML | 本仓库 |
|---|---|---|
| 语言 | Python 2 | Python 3 |
| sklearn | 0.17（用了已移除的私有 API） | 现代版（≥1.3） |
| 在线攻击 | 依赖 AWS/BigML 账号 | 本地黑盒 oracle，开箱即跑 |
| 目的 | 论文实验 | 教学 / 复现 / pre |

---

## 参考
- F. Tramèr, F. Zhang, A. Juels, M. K. Reiter, T. Ristenpart. *Stealing Machine Learning Models via Prediction APIs.* USENIX Security 2016.
- 原始代码：https://github.com/ftramer/Steal-ML
