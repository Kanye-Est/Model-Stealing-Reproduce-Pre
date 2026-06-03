# Reproduction Section 逐页演讲稿

适用文件：`slides/reproduction_section.tex` / `slides/reproduction_section.pdf`  
预计总时长：8-10 分钟  
风格：中文为主，保留 confidence、oracle、agreement、label-only 等关键术语。

---

## 第 1 页：Title

预计时间：20-25 秒

逐字稿：

接下来我进入复现部分。前面主要介绍了 Tramèr 等人在 USENIX Security 2016 这篇文章中提出的 model extraction 攻击，也就是通过 Prediction API 窃取模型。这里我不直接跑原始仓库，而是用 Python 3 重新实现几个核心攻击机制，重点看这些攻击为什么成立、需要多少 query、以及复现时有哪些算法实现上的细节。

切换提示：

先看一下这次复现采用的黑盒设置。

---

## 第 2 页：Reproduction Setting

预计时间：45 秒

逐字稿：

这一页是复现环境。我的设置是本地先训练一个 target model，记作 $f$，然后只把它包装成一个 black-box oracle。攻击者不能读取模型权重，也不能访问训练数据，只能调用类似真实 Prediction API 的接口。不同实验里，这个接口会返回 confidence、class label，或者 decision tree 的 leaf id。攻击目标是构造一个 clone model，记作 $f'$，让它和原模型 $f$ 在测试集或随机采样空间上尽量一致。

右边放的是原作者 slides 里的 generic equation-solving attack，用来说明我们和论文的抽象是一致的：输入 query，拿到输出，再反推出模型。区别是我这里没有复跑 AWS 或 BigML，而是用本地 oracle 保留攻击接口，避免 Python 2、旧 sklearn、Theano 和线上账号这些工程依赖。

切换提示：

下面先列出这次具体复现了哪些攻击和数据集。

---

## 第 3 页：Experiments and Interfaces

预计时间：45 秒

逐字稿：

这张表总结了复现实验。第一类是 equation-solving，包括二分类 Logistic Regression 和多分类 softmax、OvR Logistic Regression。这里 API 返回的是 confidence。第二类是 confidence rounding，测试如果 API 把概率四舍五入，攻击效果会下降多少。第三类是 decision-tree path finding，用 Iris 数据集训练一棵小树，黑盒接口返回 leaf id。第四类是 label-only retraining，只返回类别标签，不返回概率。最后是 KLR leakage visualization，用 Digits 数据集展示 representer-style kernel model 中训练样本可能被参数泄露。

下面的指标也很重要。Test agreement 是测试集上一致率；uniform agreement 是在归一化特征空间随机采样点的一致率；TV distance 用来衡量两个概率分布的差异。

切换提示：

先从最核心、也最优雅的 equation-solving 开始。

---

## 第 4 页：Equation-Solving: Confidence Values Become Equations

预计时间：50 秒

逐字稿：

这里用二分类 Logistic Regression 说明为什么 confidence 很危险。模型输出是 $p=\sigma(w^Tx+b)$。如果 API 返回这个概率 $p$，攻击者可以对它取 logit，也就是 $\log(p/(1-p))$。取完之后，右边就变成 $w^Tx+b$，这是关于未知参数 $w$ 和 $b$ 的线性方程。

所以对于 $d$ 维特征，只需要构造 $d+1$ 个线性无关的 query，就能得到 $d+1$ 条方程，直接解出所有权重和 bias。以 Breast Cancer 数据集为例，它有 30 个特征，所以 31 次 confidence query 就足够恢复二分类 LR。

右边的流程图就是这个过程：query 输入，API 返回 confidence，然后转成 logit，最后通过 linear solve 得到 clone model。

切换提示：

接下来看看这个思路在实际复现实验中的 query 数和一致率。

---

## 第 5 页：Equation-Solving Results

预计时间：45 秒

逐字稿：

这一页是 equation-solving 的结果。二分类 LR 用 31 次 query 达到 100% agreement。Iris softmax 有 15 个参数，对应 15 次 query，也能完全恢复。Digits softmax 和 Digits OvR 都是 650 个参数，所以使用 650 次 confidence query。

右边曲线展示了 query budget 增加时 agreement 的变化。可以看到接近 one query per parameter 时，一致率基本达到 100%。这和论文里的核心结论一致：如果 API 返回高精度 confidence，模型提取不再像普通机器学习那样需要大量带噪声样本，而更像是在解一个由 API 泄露出来的方程系统。

切换提示：

自然的防御想法是：如果 confidence 不给那么精确，会不会好一些？

---

## 第 6 页：Confidence Rounding

预计时间：40 秒

逐字稿：

这一页测试 confidence rounding。也就是 API 不返回完整概率，而是只保留若干小数位。结果可以看到，保留 5 位、4 位甚至 3 位时，test agreement 仍然是 100%，uniform agreement 也非常接近 100%。到 2 位小数时，一致率才有更明显下降，但仍然在 99% 左右。

右边曲线里，蓝线是 agreement，红线是 TV distance。随着小数位减少，概率误差会上升，但模型的分类行为并没有立刻崩掉。所以 rounding 可以增加攻击难度，但不是根本防御；真正危险的是 API 仍然在持续泄露 confidence 信息。

切换提示：

接下来换一个模型类型，看看 decision tree 怎么被提取。

---

## 第 7 页：Decision Trees: Differential Testing with Leaf IDs

预计时间：55 秒

逐字稿：

决策树和 Logistic Regression 不一样，它不是一个连续可微的概率函数，所以不能直接套 equation-solving。原论文的关键观察是：很多树模型 API 会返回由叶子节点样本分布计算出的 confidence，这个 confidence 往往可以当成 leaf id，也就是叶子的身份标识。

左边是原作者 slides 中的 decision tree attack。思想是 differential testing：构造两个输入 $x$ 和 $x'$，让它们只在一个特征上不同。如果输出的 leaf id 变了，就说明路径上某个判断和这个特征有关。

我的复现里，先 query 一个点记录 leaf id，然后逐个改变特征；对连续特征，用 binary search 找阈值；发现新的区域后继续递归。Iris 上目标树有 5 个叶子，最终也提取出 5 个叶子，557 次 query，test 和 uniform agreement 都是 100%。

切换提示：

不过这个算法实现时有一个容易踩的坑，下一页单独说明。

---

## 第 8 页：Decision Tree Implementation Detail

预计时间：55 秒

逐字稿：

这里是复现 decision tree 时最值得讲的实现细节。一个天真的做法是：对某个特征区间，只检查左右两个端点的 leaf id。如果端点 leaf id 一样，就认为中间没有分裂。但这个假设是错的。

左边图里，左右端点都落在 leaf A，但中间有一段区域会落到 leaf B。也就是说，端点相同不代表区间内部没有隐藏叶子。如果只看端点，提取器会漏掉中间这段路径。

所以我的实现不是简单比较区间两端，而是围绕当前叶子的路径谓词逐步恢复每个特征上的约束，并显式维护 interval constraints。连续阈值还要处理 $\leq$ 和 $>$ 的边界问题。这个细节解释了为什么复现不是简单跑一个树遍历，而是要把论文里的 path finding 思路落实到可执行的谓词恢复过程。

切换提示：

下一页看提取出来的树和原树的对应关系。

---

## 第 9 页：Extracted Tree: Larger View

预计时间：35 秒

逐字稿：

这一页把决策树结果放大。左侧是本地训练出来的 sklearn target tree，右侧是黑盒提取后得到的 leaf intervals。这里攻击者没有访问树结构，只通过 leaf id query 反推出每个叶子对应的特征区间和预测类别。

这张图的重点不是树本身很复杂，而是说明即使是离散的、分段的模型，只要 API 暴露了稳定的叶子身份信息，攻击者仍然可以把模型的决策区域一点点拼出来。

切换提示：

如果进一步限制 API，只返回 label，不返回 confidence 或 leaf id，会发生什么？

---

## 第 10 页：Label-Only Extraction

预计时间：50 秒

逐字稿：

这一页对应论文中的 label-only setting。也就是 API 只返回类别标签，不返回 confidence。左边是原作者 slides 里的 active retraining 思路：攻击者不断在当前 clone model 的 decision boundary 附近 query，因为边界附近的信息量最大。

右边是我的复现结果。对 Digits softmax，如果仍然有 confidence，650 次 query 就能 100% 提取；但如果只返回 label，650 次 uniform retraining 只有 68.83% uniform agreement，adaptive retraining 也只有 76.13%。当 query 增加到 6500 次和 26000 次时，adaptive 方法逐步接近目标模型，最后达到 99.33%。

这说明 label-only 是有效的 API minimization，它显著提高攻击成本；但它不是绝对防御，因为主动学习式 retraining 仍然可以逼近模型。

切换提示：

下一页把 confidence 和 label-only 的 query cost 放在一张图里对比。

---

## 第 11 页：Confidence vs Label-Only Query Cost

预计时间：35 秒

逐字稿：

这张图更直观地展示了成本差异。红色星号是 confidence equation-solving：650 次 query 就达到接近 100% agreement。蓝线和橙线是 label-only 的 uniform 和 adaptive retraining，横轴是 log scale 的 query 数。

可以看到，在同样 650 次 query 下，label-only 明显差很多；但 query 数增加两个数量级后，adaptive retraining 仍然能追上来。这和论文结论一致：隐藏 confidence 可以把攻击从“解方程”推回“学习问题”，但不能完全消除 model extraction。

切换提示：

除了偷模型功能，论文还强调模型提取可能带来隐私风险。下面看 KLR 的例子。

---

## 第 12 页：KLR Leakage: Parameters Can Contain Training Samples

预计时间：50 秒

逐字稿：

这一页是 KLR leakage visualization。这里我实现的是一个教学版 representer model：每个类别的 score 由若干 representer 和输入之间的 RBF 相似度累加得到。直观上，representer 就是模型参数的一部分，但它们本身可能非常接近训练样本。

我在 Digits 数据集上每个类别选 5 个 representer，总共 50 张图。右边展示的是原始 representer 和提取出来的 representer。可以看到，它们就是可视化的手写数字图像。

这个例子的意义是：model extraction 不一定只偷走模型功能。对于某些模型结构，参数本身可能携带训练数据的内容，所以提取模型也可能成为训练数据泄露的入口。

切换提示：

最后总结一下这次本地复现支持了哪些论文结论。

---

## 第 13 页：Reproduction Takeaways

预计时间：50 秒

逐字稿：

总结一下这次复现。第一，confidence API 会让 Logistic Regression 的提取从普通学习问题变成 system-solving problem。第二，多分类 LR 在复现中也能达到 roughly one query per parameter 的效果。第三，decision tree 的提取依赖 leaf identity，而且实现时必须认真处理路径谓词和区间边界。第四，label-only API 能显著提高 query cost，但不能完全阻止 extraction。第五，representer-style kernel model 展示了模型参数泄露训练样本的风险。

这次没有复跑 AWS 和 BigML 在线实验，因为重点是做一个本地、可审计、可展示的攻击机制复现。整体结论和论文主线是一致的：越丰富的 Prediction API，尤其是 confidence 和部分结构信息，越容易和模型及数据的机密性发生冲突。

切换提示：

复现部分到这里结束，后面可以进入讨论：API 应该返回多少信息，以及 model extraction 在今天的大模型服务里有什么对应问题。
