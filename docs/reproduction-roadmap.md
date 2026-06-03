# 复现路线图

目标：用现代 Python 3 在本地复现论文核心现象，不依赖 AWS / BigML 账号。所有实验都采用“本地训练受害者模型，但只通过 oracle 查询接口攻击”的方式。

---

## 0. 复现原则

1. **不 fork 原仓库**：原始 Steal-ML 是 Python 2 研究代码，本仓库重写核心算法。
2. **先复现现象，再追求完全数值一致**：论文数据和服务行为已过时，重点是查询量级、趋势和攻击成功机制。
3. **统一 oracle 抽象**：攻击代码只能调用 `query_label(x)` / `query_proba(x)` / `query_leaf_id(x)`，不能读取模型参数。
4. **统一指标**：
   - `agreement_test`：测试集上副本与 oracle 标签一致率。
   - `agreement_uniform`：均匀采样输入空间上一致率。
   - `tv_distance`：概率输出的 total variation 距离。
   - `query_count`：oracle 查询次数。

---

## Phase 1：二元 Logistic Regression 方程求解

### 对应论文

- §4.1 Binary logistic regression
- slides p8-p10

### 复现内容

训练二元 LR 作为黑盒：

```text
p = sigmoid(w·x + b)
logit(p) = w·x + b
```

攻击：

1. 随机生成 `d + 1` 个线性无关 query。
2. 查询概率 `p`。
3. 解线性系统 `[X, 1] @ theta = logit(p)`。
4. 得到副本模型。

### 成功标准

- `query_count = d + 1`
- `agreement_test = 100%`
- `agreement_uniform = 100%`
- `tv_distance < 1e-8`

### 组会展示

现场最适合跑这个实验。它只需要秒级运行，并且能直观看出“置信度把分类问题变成解方程问题”。

---

## Phase 2：Softmax / OvR 方程求解

### 对应论文

- §4.1 Multiclass LRs and MLPs
- Table 4

### 复现内容

训练多类 LR：

- softmax
- one-vs-rest

攻击：

1. 随机生成 query。
2. 查询完整概率向量。
3. 用 soft labels 最小化 cross-entropy。
4. 得到同结构副本。

### 实现建议

不要使用原仓库旧 sklearn 私有函数。直接手写：

- `softmax(logits)`
- `cross_entropy(P_oracle, P_clone)`
- 梯度，或用 `scipy.optimize.minimize`

### 成功标准

- query 量级约等于参数量 `c * (d + 1)`。
- 测试集和均匀采样一致率接近 100%。
- 概率 TV 距离很小。

### 注意点

softmax 参数有平移不唯一性，所以不要用参数 L2 误差作为主指标。

---

## Phase 3：置信度 rounding 防御

### 对应论文

- §7 Rounding confidences

### 复现内容

在 Phase 2 的 oracle 上加一层包装：

```python
p = np.round(p, digits)
p = p / p.sum(axis=1, keepdims=True)
```

测试 `digits = 5, 4, 3, 2` 对攻击的影响。

### 成功标准

趋势应接近论文：

- 4-5 位：基本不影响。
- 2-3 位：攻击变弱，但通常仍强于 label-only retraining。

---

## Phase 4：决策树 Path-Finding

### 对应论文

- §4.2 Decision Tree Path-Finding
- Algorithm 1
- slides p14

### 复现内容

训练 `sklearn.tree.DecisionTreeClassifier`，包装为黑盒 oracle：

- `query_label(x)`
- `query_leaf_id(x)`：返回 `(label, confidence)` 或内部 leaf id 的教学模拟版本
- 统计查询次数

第一版 bottom-up：

1. 随机完整 query。
2. 对每个特征做差分测试。
3. 连续特征二分找阈值。
4. 类别特征逐值测试。
5. 记录每个 leaf 的谓词路径。

第二版 top-down：

模拟 partial query：当缺少当前 split feature 时，oracle 停在当前节点并返回 node id。这个版本用于复现“不完整查询显著降低查询数”的结论。

### 成功标准

- 能恢复所有叶子路径，或在测试/均匀采样上达到 100% 一致。
- 输出原树叶子数、恢复叶子数、query_count。

### 组会展示

建议准备一个 2D toy tree 图和一个真实小数据集结果。toy tree 用来讲算法，真实结果用来证明可运行。

---

## Phase 5：Label-Only 攻击

### 对应论文

- §5 Extraction Given Class Labels Only
- slides p15-p16

### 复现内容

对同一个二元 / 多类 LR oracle，隐藏概率，只返回标签。

实现三种策略：

| 策略 | 说明 |
|---|---|
| uniform retraining | 随机采样点，查询标签，本地重训 |
| line-search retraining | 找正负 / 不同类样本之间的边界点 |
| adaptive retraining | 先训副本，再查询当前副本最不确定的点 |

### 成功标准

- adaptive 优于 uniform。
- 达到高一致率所需查询数明显高于 equation-solving。
- 能讲清“去掉置信度能提高成本，但不是根本防御”。

---

## Phase 6：KLR 训练数据泄露可视化

### 对应论文

- §4.1.3 Training Data Leakage for Kernel LR
- Figure 5

### 复现内容

教学版目标：

1. 用 digits 数据训练 RBF-kernel 风格概率模型。
2. representers 选自训练样本。
3. 攻击端尝试恢复或近似 representers。
4. 输出 `figures/klr_leakage.png`：原 representers vs 提取结果。

### 成功标准

视觉上能看出训练样本轮廓，或者能展示低估 representer 数量时的类平均图像泄露。

### 优先级

这部分适合作为 pre 加分项，但实现复杂度高于 LR 和 tree，不应阻塞主线。

---

## 建议命令设计

未来实现后建议保留这些入口：

```bash
python -m src.equation_solving.run_binary_lr
python -m src.equation_solving.run_multiclass_lr
python -m src.equation_solving.run_rounding
python -m src.decision_tree.run_tree_extraction
python -m src.label_only.run_label_only
python -m src.klr_leakage.run_klr_leakage
```

输出约定：

```text
experiments/results/*.csv
figures/*.png
docs/reproduction-report.md
```

---

## 最小可交付版本

如果组会时间紧，最小版本做这三项即可：

1. 二元 LR：`d + 1` 次查询精确提取。
2. Softmax：约 `1 query / parameter` 提取。
3. Label-only adaptive：隐藏置信度后查询数上涨约两个数量级。

这三项已经能完整支撑论文主结论。决策树和 KLR 可以作为扩展演示。

