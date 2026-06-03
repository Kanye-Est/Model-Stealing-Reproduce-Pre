# 作者 USENIX'16 Slides 借鉴笔记

本地文件：`../security16_slides_tramer.pdf`

作者 slides 总共 17 页，叙事非常适合组会 pre：先讲 MLaaS 的商业与安全矛盾，再用 LR 一页推导核心攻击，随后扩展到 AWS、决策树、防御和结论。我们的 pre 可以沿用这个节奏，并在最后加入 Python 3 复现结果。

---

## 1. 作者 slides 的叙事结构

| 页码 | 内容 | 组会借鉴方式 |
|---|---|---|
| p1 | 标题 | 保留论文信息，说明 USENIX Security 2016 |
| p2 | ML 系统流程：数据、训练、预测 | 用作背景图，快速解释 `f(x)=y` |
| p3 | MLaaS 核心矛盾：rich API vs confidentiality | 这是开场核心页，建议重点讲 |
| p4 | 不同 MLaaS 服务和模型类型 | 可以压缩成一页表格 |
| p5 | Model extraction 定义与动机 | 讲清攻击目标：`f'(x)=f(x)` |
| p6 | 为什么不是“普通机器学习” | 强调现实 API 返回更多信息 |
| p7 | Main results | 用作贡献总览 |
| p8-p9 | Logistic Regression 示例和 logit 推导 | 技术主菜，建议保留手推 |
| p10 | Generic equation-solving attacks | 从二元 LR 推广到 MLR/NN |
| p11-p12 | AWS case study | 讲特征提取逆向和真实服务成本 |
| p13 | Extract + model inversion | 讲隐私危害 |
| p14 | Decision tree extraction | 讲 confidence 作为 leaf pseudo-id |
| p15-p16 | Countermeasure: label-only 与 active retraining | 讲防御不能完全阻止 |
| p17 | Conclusion | 结束总结 |

---

## 2. 最值得复用的三页

### p3：Rich Prediction APIs vs Model Confidentiality

这页是全文问题意识：

- MLaaS 希望 API 友好：高可用、高精度、返回 confidence。
- 模型拥有者希望模型和训练数据保密。
- 二者冲突：API 返回的信息越多，越方便攻击者重建模型。

组会讲法：

> 这篇文章不是在问“机器学习模型能不能被学习”，而是在问：当一个商业预测 API 为了好用而返回高精度概率时，它是不是已经把模型参数用方程的形式泄露出来了？

### p8-p10：LR 到方程求解

必须作为技术重点：

```text
f(x) = 1 / (1 + exp(-(w*x + b)))
log(f(x) / (1 - f(x))) = w*x + b
```

讲法顺序：

1. 二元 LR 有 `n + 1` 个未知数。
2. 每个 confidence query 给一个线性方程。
3. 查询 `n + 1` 个线性无关点就能解出参数。
4. 多类 LR / MLP 没有解析线性解，但可以把 oracle 概率当作 soft label 做“无噪声训练”。

这几页比论文文字更适合口头解释，建议照着动画化推导。

### p14：Decision Tree

核心类比：

- LR 的 confidence 是方程右端。
- Tree 的 confidence 是叶子路径的“身份证”。

讲法：

> 对树来说，输出不是连续函数，不能再解方程。但如果每个叶子的 confidence 基本唯一，那我们每次改一个特征，就能知道自己是不是走到了另一片叶子。于是 confidence 从概率变成了路径标识符。

---

## 3. 我们的 pre 建议页序

建议 20-25 分钟正文 + 5-8 分钟复现：

| 页 | 标题 | 内容 |
|---|---|---|
| 1 | 标题 | 论文、作者、会议、代码仓库 |
| 2 | MLaaS 背景 | 训练 API / prediction API / pay per query |
| 3 | 核心矛盾 | rich API vs confidentiality |
| 4 | 攻击定义 | 黑盒查询，目标 `f' ≈ f` |
| 5 | 为什么不是普通学习 | confidence + partial query |
| 6 | 贡献总览 | equation-solving / path-finding / label-only |
| 7 | 二元 LR 推导 | logit 变线性方程 |
| 8 | 方程求解泛化 | softmax / OvR / MLP，1 query per parameter |
| 9 | 真实 AWS 案例 | feature extraction reverse engineering |
| 10 | 隐私危害 | KLR 或 extract-and-invert |
| 11 | 决策树攻击 | confidence as pseudo-id |
| 12 | BigML 结果 | German Credit 等查询数 |
| 13 | Label-only 防御 | Lowd-Meek + adaptive retraining |
| 14 | Countermeasures | rounding / DP / ensemble |
| 15 | 我们的复现设计 | 本地 oracle，不依赖在线账号 |
| 16 | 复现实验结果 | LR / softmax / label-only / tree |
| 17 | 总结与讨论 | LLM/API 时代延伸 |

---

## 4. 现场演示优先级

### 必做演示：二元 LR

原因：

- 运行快。
- 公式最清楚。
- 查询数 `d + 1` 很有冲击力。

现场输出建议：

```text
target dim: 30
queries: 31
test agreement: 1.0000
uniform agreement: 1.0000
mean TV distance: 2.1e-10
```

### 推荐演示：label-only 对比

展示同一个 oracle：

- 返回 confidence：几十 / 几百次查询直接提取。
- 只返回 label：需要更多查询，adaptive 才逐渐接近。

这能自然过渡到防御讨论。

### 可选演示：决策树 toy example

如果现场时间有限，用静态图 + 已跑结果即可，不必现场跑完整树提取。

---

## 5. 可以借用的作者表达

建议保留这些概念表达：

| 作者表达 | 中文讲法 |
|---|---|
| Rich Prediction APIs | 信息丰富的预测接口 |
| Model Confidentiality | 模型机密性 |
| Noiseless Machine Learning | 无噪声机器学习 |
| Pseudo-identifier | 伪标识符 / 路径身份证 |
| API Minimization | API 最小化 |
| Extract-and-Invert | 先提取、再反演 |

---

## 6. 不建议照搬的地方

1. **不要花太久列所有 MLaaS 服务**：2016 年服务形态已经过时，只用它说明当时现实 API 的普遍性。
2. **不要现场讲太多 KLR 公式**：用图讲“representers 是训练点”更有效。
3. **不要把防御讲成已解决问题**：论文结论是 API minimization 能提高成本，但 label-only 仍可被攻击。
4. **不要承诺原 AWS/BigML 实验可复跑**：服务行为、账号接口和价格都可能变化。本仓库复现的是本地 oracle 版本。

