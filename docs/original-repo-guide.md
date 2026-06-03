# 原始 Steal-ML 仓库导读

原始仓库：`git@github.com:ftramer/Steal-ML.git`

本地位置：`../Steal-ML/`

结论先说：原仓库非常适合作为算法参考，但不适合直接作为本仓库的运行基础。它是 2016 年左右的 Python 2 研究代码，依赖旧版 `scikit-learn` 私有 API、Theano、BigML/AWS 在线账号接口和若干硬编码路径。后续复现建议采用“读原实现，Python 3 重写离线黑盒 oracle”的路线。

---

## 1. 顶层结构与论文对应关系

| 原仓库目录 | 对应论文内容 | 作用 | 复现建议 |
|---|---|---|---|
| `regression/` | §4.1 Equation-Solving；§7 rounding；Amazon LR 相关代码 | LR / softmax / OvR / KLR 的方程求解攻击 | 第一优先级，按算法重写 |
| `trees/` | §4.2 Decision Tree Path-Finding；BigML 实验 | 决策树路径查找、top-down 变体、BigML wrapper | 第二优先级，重写本地 sklearn tree oracle |
| `binary-classifiers/` | §5 Label-Only；Lowd-Meek；SVM retraining | 只返回标签时的线搜索、主动学习、SVM 复训 | 第三优先级，选精简版实现 |
| `neural-nets/` | §4.1 MLP equation-solving；§5 NN label-only | Theano MLP 训练和提取 | 可选，现代版用 sklearn/PyTorch 替代 |
| `data/` | 论文公开数据集子集 | Adult、Steak、GSS、Iris 等 | 可复用小数据集；大数据需重新整理 |
| `sec16_paper.pdf` | 会议论文 PDF | 原论文 | 已有本地 PDF/LaTeX，可交叉核对 |

---

## 2. `regression/`：方程求解攻击主线

关键文件：

| 文件 | 内容 |
|---|---|
| `regression_stealer.py` | 核心抽象 `RegressionExtractor`，包含概率查询、优化求解、二元 LR 线性求解、adaptive 采样 |
| `regression.py` | 本地 sklearn logistic regression 黑盒，直接在特征空间查询 |
| `regression_cat.py` | 带 one-hot encoding 的输入空间查询版本 |
| `kernel_regression.py` / `kernel_regression_stealer.py` | KLR 与 representer 泄露实验，依赖 Theano |
| `aws_wrapper/` | Amazon ML 的在线攻击、特征逆向和数据准备 |
| `Makefile` | 批量跑 binary / softmax / OvR / rounding 实验 |

### 2.1 二元 LR 的精确求解

原实现位置：`RegressionExtractor.find_coeffs_bin`

算法：

1. 生成 `budget` 个查询点 `X_train`。
2. 取目标模型返回概率 `p = query_probas(X_train)[:, 1]`。
3. 计算 `logit(p)`。
4. 拼接偏置列 `[X, 1]`。
5. 当 `budget == d + 1` 时用 `np.linalg.solve`；否则用最小二乘。

这是本仓库最应该先复现的部分，因为它和论文公式完全一致，工程风险最低，也最适合组会现场演示。

### 2.2 多类 LR 的“无噪声训练”

原实现位置：

| 函数 | 作用 |
|---|---|
| `logistic_loss` / `logistic_grad` | OvR 风格概率的交叉熵和梯度 |
| `multinomial_loss` / `multnomial_grad` | softmax 交叉熵，调用旧版 sklearn 私有函数 |
| `select_and_run_opti` | 根据 softmax / OvR / binary 选择求解器 |
| `run_opti` | BFGS / L-BFGS-B 多轮优化 |

注意事项：

- `sklearn.linear_model.logistic._multinomial_loss` 等私有 API 已在现代 sklearn 中移除，不能直接迁移。
- Python 3 版本应自己写 softmax cross-entropy，或者用 `scipy.optimize.minimize` + 手写梯度。
- softmax 参数有平移不唯一性，复现评价应以功能一致率和概率 TV 距离为准，不要求参数逐项相等。

### 2.3 rounding 防御实验

原实现位置：`LocalRegressionExtractor.query_probas`

逻辑很简单：对概率 `np.round(p, digits)` 后重新归一化。论文结论是 4-5 位基本无影响，2-3 位会削弱但仍明显优于 label-only。

### 2.4 KLR 训练数据泄露

原实现依赖 Theano，直接迁移成本较高。复现目标不必一开始完全等价，可以先做教学版：

1. 用 digits 训练一个 RBF kernel logistic / kernel ridge classifier 风格模型。
2. 让 representers 是训练样本子集。
3. 攻击端学习 representer 坐标或展示低估 representer 数量时的类均值泄露。

这部分在 pre 中视觉冲击很强，但实现优先级低于二元/多类 LR。

---

## 3. `trees/`：决策树路径查找

关键文件：

| 文件 | 内容 |
|---|---|
| `tree_stealer.py` | 核心 `TreeExtractor`，实现 bottom-up path-finding 和 top-down 变体 |
| `feature.py` | 连续 / 类别特征抽象 |
| `predicate.py` | `<=`、`>`、类别集合谓词及合并逻辑 |
| `bigml_wrapper/tree.py` | BigML 模型解析、在线/本地查询封装 |
| `Makefile` | BigML public model 批量实验 |

### 3.1 bottom-up path-finding

原实现位置：`TreeExtractor.extract`

核心流程：

1. 随机生成一个完整 query。
2. 查询得到 leaf id。
3. 对每个特征做差分测试：
   - 连续特征：`test_cont_feature` -> `line_search`
   - 类别特征：`test_cat_feature`
4. 得到当前叶子的路径谓词。
5. 构造能走向其他叶子的 query，加入队列。
6. 队列清空时得到所有叶子路径。

leaf id 在 BigML 中由 `(prediction, confidence, fields)` 近似构成。论文强调 confidence 是路径伪标识符；代码中还把 fields 放进 id 以支持 top-down。

### 3.2 top-down 变体

原实现位置：`TreeExtractor.extract_top_down`

它利用不完整查询，从空 query 开始，逐层找当前节点的 split feature。论文和 slides 都强调这个版本在 BigML 上查询数更少，例如 German Credit 从 1,722 降到 1,150。

本仓库离线复现建议：

- 第一版先实现完整 query 的 bottom-up，避免处理缺失值语义。
- 第二版再模拟 partial query：当 query 缺少某个 split feature 时，让 oracle 停在当前节点并返回 node id。

### 3.3 原代码的迁移问题

- `iteritems`、`xrange`、`print` 均为 Python 2 写法。
- BigML wrapper 依赖在线服务和 public model 元数据，今天不保证可用。
- `bigml_wrapper/tree.py` 中 `asdf` 是故意留下的调试中断，说明这不是稳定 CLI。

所以树攻击应重写为纯本地版本：训练 `sklearn.tree.DecisionTreeClassifier`，自己暴露一个只返回 leaf id / label / confidence 的 oracle。

---

## 4. `binary-classifiers/`：Label-Only 与 SVM

关键文件：

| 文件 | 内容 |
|---|---|
| `algorithms/OnlineBase.py` | 随机采样正负点、二分推到决策边界 |
| `active_learning.py` | RBF SVM 主动学习式 retraining |
| `LowBudgetComparison.py` | 低预算 RBF SVM 复训对比 |
| `cracksvm.py` | libsvm 模型 oracle + 迭代采边界点 + 重训 |
| `algorithms/LinearTrainer.py` | 线性模型训练/搜索 |
| `algorithms/RBFTrainer.py` | RBF retraining |

### 4.1 Lowd-Meek / boundary sampling

原实现位置：`OnlineBase.push_to_b`

核心是找到一正一负两个点，然后二分到决策边界附近。和论文 §5 的 Lowd-Meek 直觉一致。

### 4.2 Adaptive retraining

原仓库中有两类实现：

- LR 版本在 `RegressionExtractor.find_coeffs_adaptive`：先训练当前副本，再找当前模型最不确定的点去问 oracle。
- SVM 版本在 `active_learning.py` / `RBFTrainer.py`：围绕边界采点并重训 SVM。

本仓库建议先实现 LR label-only：

1. `UniformRetraining`
2. `BoundaryLineSearchRetraining`
3. `AdaptiveRetraining`

等组会材料够用后，再扩展 RBF SVM。

---

## 5. `neural-nets/`：MLP

原仓库用 Theano 写了一层 hidden layer + softmax 的 MLP，并通过 soft labels 做提取。

迁移建议：

- 不迁移 Theano。
- 若需要展示 MLP，可用 `sklearn.neural_network.MLPClassifier` 或 PyTorch。
- pre 中不需要现场完整跑 MLP，论文数值可作为扩展结果讲；复现主线用 LR + tree + label-only 已足够说明贡献。

---

## 6. 原仓库运行风险清单

| 风险 | 具体表现 |
|---|---|
| Python 2 | `print` 语句、`xrange`、`iteritems`、`cPickle` |
| 旧 sklearn | `sklearn.cross_validation`、`sklearn.grid_search`、`sklearn.linear_model.logistic._multinomial_loss` |
| Theano 停维护 | KLR / NN 依赖 Theano，现代 Python 安装成本高 |
| 在线服务依赖 | AWS / BigML API、账号、模型 ID、服务行为可能已变化 |
| 硬编码路径 | 如 `HOME/Dropbox/Projects/SVM/...` |
| 研究代码调试残留 | `asdf`、手动 Makefile 输出、非稳定 CLI |

---

## 7. 本仓库实现优先级

建议按组会收益排序：

1. **二元 LR equation-solving**：最短代码、最清楚公式、最适合现场演示。
2. **Softmax equation-solving**：复现论文“1 query / parameter”结论。
3. **Decision tree bottom-up path-finding**：展示 confidence 作为 leaf id 的第二个核心洞察。
4. **Label-only adaptive retraining**：讲防御后仍可攻击。
5. **KLR leakage 可视化**：有时间再做，视觉效果最好。
6. **MLP / RBF SVM**：作为扩展实验，不阻塞 pre。

