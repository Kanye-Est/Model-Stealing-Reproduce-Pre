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

---

## Phase 3：置信度 rounding 防御实验

对应论文：§7 Rounding confidences。

实验入口：

```bash
python -m src.equation_solving.run_rounding
python -m src.equation_solving.run_rounding --model ovr
```

实验设置：

- 数据集：`sklearn.datasets.load_digits`
- 受害者模型：softmax / OvR 多类 LR
- 攻击预算：650 次查询，即 `1.0 * c * (d + 1)`
- 攻击者看到的概率：先按 `decimals` 四舍五入，再重新归一化
- 评估基准：未取整的真实 oracle 标签和概率
- 均匀评估点：10,000

当前结果（Python 3.14.5，seed=0，Digits）：

| 模型 | 置信度小数位 | 查询数 | 测试集一致率 | 均匀一致率 | 测试 TV | 均匀 TV | 优化成功 |
|---|---:|---:|---:|---:|---:|---:|---|
| softmax | 5 | 650 | 100.0000% | 99.9700% | `1.601e-06` | `7.122e-06` | True |
| softmax | 4 | 650 | 100.0000% | 99.9600% | `1.962e-05` | `6.624e-05` | True |
| softmax | 3 | 650 | 100.0000% | 99.9000% | `2.749e-04` | `7.964e-04` | True |
| softmax | 2 | 650 | 99.6296% | 99.0900% | `3.170e-03` | `1.068e-02` | True |
| OvR | 5 | 650 | 100.0000% | 100.0000% | `1.343e-04` | `3.027e-05` | True |
| OvR | 4 | 650 | 100.0000% | 99.9600% | `4.403e-04` | `2.642e-04` | True |
| OvR | 3 | 650 | 100.0000% | 99.7900% | `4.508e-03` | `2.899e-03` | True |
| OvR | 2 | 650 | 98.5185% | 98.3000% | `4.875e-02` | `1.964e-02` | False |

结论：复现了论文中 rounding 防御的核心趋势。4-5 位小数几乎不影响模型提取；3 位会增加概率误差但标签一致率仍很高；2 位才显著削弱攻击，但在 Digits 上仍能得到 98%-99% 左右的功能一致率。因此，置信度取整可以提高攻击难度，但不是完整防御。

实现备注：`run_rounding` 使用 `RoundedProbaOracle` 包装真实 oracle，只改变攻击者看到的 `query_proba` 输出；评估时仍与未取整的真实模型比较。

---

## Phase 4：决策树 Path-Finding（bottom-up 教学版）

对应论文：§4.2 Decision Tree Path-Finding。

实验入口：

```bash
python -m src.decision_tree.run_tree_extraction
```

实验设置：

- 数据集：`sklearn.datasets.load_iris`
- 特征缩放：`[-1, 1]`
- 受害者模型：`sklearn.tree.DecisionTreeClassifier(max_depth=3)`
- 攻击能力：
  - `query_leaf_id(x)`：返回 sklearn leaf index，作为论文中 leaf identity oracle 的本地教学版
  - `query_label(x)`：只用于记录叶子预测标签和评估
- 攻击方式：bottom-up path finding。对当前 query 到达的叶子，逐特征做一维差分测试和二分，找到仍通向当前 leaf id 的连续区间；其它区间生成新 query 去探索未访问叶子。
- 连续特征精度：`epsilon=1e-5`
- 均匀评估点：10,000

当前结果（Python 3.14.5，seed=0）：

| 指标 | 数值 |
|---|---:|
| 特征维度 `d` | 4 |
| 目标树叶子数 | 5 |
| 提取叶子数 | 5 |
| 目标树深度 | 3 |
| 提取查询数 | 557 |
| 测试集一致率 | 100.0000% |
| 均匀采样一致率 | 100.0000% |

结论：复现了决策树 path-finding 的核心机制：攻击者不读取树结构，只用 leaf identity oracle，通过逐特征差分测试恢复每个叶子的路径谓词，并得到与目标树功能一致的本地副本。

实现备注：当前版本是教学版，覆盖连续特征和唯一 leaf id；尚未实现论文中 BigML 风格的 confidence pseudo-identifier、categorical split、duplicate id 处理和 partial-query top-down 变体。

---

## Phase 5：Label-Only retraining 攻击

对应论文：§5 Extraction Given Class Labels Only。

实验入口：

```bash
python -m src.label_only.run_label_only
```

实验设置：

- 数据集：`sklearn.datasets.load_digits`
- 受害者模型：softmax 多类 LR
- 攻击能力：只能调用 `query_label(x)`，看不到 confidence values
- 对比策略：
  - uniform retraining：均匀随机采样 query，查询标签后本地重训 softmax LR
  - adaptive retraining：先用少量均匀 query 训练副本，再反复在当前副本最不确定的点附近查询标签并重训
- 均匀评估点：10,000

当前结果（Python 3.14.5，seed=0，Digits softmax）：

| 策略 | 查询预算 | 测试集一致率 | 均匀一致率 |
|---|---:|---:|---:|
| uniform | 650 | 94.4444% | 68.8300% |
| adaptive | 650 | 95.0000% | 76.1300% |
| uniform | 6,500 | 99.6296% | 94.3700% |
| adaptive | 6,500 | 100.0000% | 97.8300% |
| uniform | 26,000 | 99.6296% | 97.7300% |
| adaptive | 26,000 | 100.0000% | 99.3300% |

结论：复现了论文中 label-only 防御下的核心趋势。隐藏 confidence 后，650 次查询不再足以在整个输入空间上精确提取；查询预算提高到 6,500 / 26,000 后，副本才逐步接近目标模型。adaptive retraining 在相同预算下明显优于 uniform retraining，但仍比 confidence-based equation-solving 昂贵得多。

实现备注：当前版本实现 uniform 与 adaptive retraining；尚未实现 Lowd-Meek 线性边界恢复和 line-search retraining。

---

## Phase 6：KLR representer 训练数据泄露可视化

对应论文：§4.1.3 Training Data Leakage for Kernel LR。

实验入口：

```bash
python -m src.klr_leakage.run_klr_leakage
```

实验设置：

- 数据集：`sklearn.datasets.load_digits`
- 模型：RBF representer classifier，用 KLR 风格的 representer 矩阵计算类别概率
- representer 选择：每个数字类别随机选 5 张训练图像，共 50 个 representers
- 攻击展示：一旦模型被提取为白盒，representer 矩阵本身就是模型参数的一部分，因此训练图像子集随模型一起泄露
- 输出图片：`figures/klr_leakage.png`

当前结果（Python 3.14.5，seed=0）：

| 指标 | 数值 |
|---|---:|
| 类别数 | 10 |
| 每类 representer 数 | 5 |
| 总 representer 数 | 50 |
| gamma | 0.08 |
| 目标模型测试准确率 | 78.7037% |
| 输出图 | `figures/klr_leakage.png` |

结论：复现了论文中 KLR 泄露现象的教学版：对于显式保存训练样本子集作为 representers 的 kernel 模型，模型提取不仅泄露决策函数，也会泄露嵌入模型参数中的训练图像。

实现备注：当前版本强调“representers 随白盒模型泄露”的可视化，不实现论文中通过优化反推出 representer 坐标的完整攻击。
