# MovieLens 1M 电影推荐系统

CS5481 小组项目一。实现数据探索、预处理，以及 ItemCF、LightFM 和 LRURec 的训练、评估与推荐。

## 环境与运行

可使用预编译的 Conda 环境，在项目根目录执行（使用独立名称避免与已有环境混用）：

```bash
conda env create -n movie-lightfm -f environment.yml
conda activate movie-lightfm
python -m jupyterlab
```

其他环境可使用 Python 3.10 或更高版本安装 `requirements.txt`（LightFM 从源码安装需要 C 编译器）：

```bash
python -m pip install -r requirements.txt
python -m jupyterlab
```

打开 `notebooks/01_data_analysis.ipynb`，选择安装了依赖的 Python 内核，
按顺序运行全部单元格。在 VS Code 中也可以直接打开该 Notebook 并选择相同环境。

## 数据准备

从 [GroupLens 官方页面](https://grouplens.org/datasets/movielens/1m/)
下载 MovieLens 1M，解压并阅读附带的 `README`。读取器自动检查以下目录：

1. `data/ml-1m/`
2. `ml-1m/ml-1m/`（当前本地数据的位置）
3. `ml-1m/`

目录中需同时包含 `users.dat`、`movies.dat`、`ratings.dat`。
也可以通过 `load_movielens(data_dir="自己的目录")` 指定路径；
相对路径以项目根目录为基准，绝对路径直接使用。
原始数据仅保留在本地，不纳入 Git；每位组员分别从官方渠道下载。

## 文件职责

```text
src/data.py                         数据读取、质量检查、清洗、时间划分与相关性标签
src/features.py                     训练期特征、稀疏矩阵、行为序列与共享评估上下文
src/models/item_cf.py               正反馈 ItemCF、模型保存与恢复
src/models/lightfm.py               LightFM 特征编码、向量导出与推荐评分
src/models/lrurec.py                LRURec 线性递推模型、序列评分与保存恢复
src/sequences.py                    下一部电影训练窗口与序列补齐
src/train_lrurec.py                 LRURec 训练、验证选轮数与共享评估
src/train_lightfm.py                官方 LightFM 训练、验证选参及结果比较
src/evaluate.py                     共享排名、Precision/Recall/NDCG、命中统计与覆盖率
src/recommend.py                    电影推荐、标题/类型展示与相似历史解释
train.py                           验证集调参、选定模型的最终测试与结果导出
recommend.py                       按用户 ID 输出推荐的命令行入口
audit_itemcf.py                    原始数据对齐、相似度/评估审计及推荐观测状态诊断
notebooks/01_data_analysis.ipynb     数据概况、探索总览图与预处理
notebooks/02_itemcf.ipynb            ItemCF 原理、搜索结果、评估及用户推荐演示
notebooks/03_lightfm.ipynb           LightFM 原理、核心结果与双算法推荐比较
notebooks/04_lrurec.ipynb            LRURec 原理、核心结果与三算法推荐比较
outputs/figures/                    Notebook 自动生成的 PNG 图表
outputs/tables/                     Notebook 自动生成的统计汇总 CSV
data/processed/                    本地模型输入与配置，不纳入 Git
tests/test_preprocessing.py         时间边界、留出数据隔离和候选过滤测试
tests/test_itemcf.py                相似度手算、排名指标、推荐过滤和保存恢复测试
requirements.txt                    数据阶段的依赖
environment.yml                     Conda 环境依赖
```

读取示例（在项目根目录运行）：

```python
from src.data import load_movielens, dataset_summary

tables = load_movielens()
users = tables["users"]
movies = tables["movies"]
ratings = tables["ratings"]
print(dataset_summary(tables))
```

读取时使用 `::` 分隔符和 Latin-1 编码，邮编按字符串保留，时间戳保留为原始 Unix 秒数。
年龄是年龄段编码，职业是类别编码，电影类型暂时保留为 `|` 分隔字符串。
原始表用于探索；预处理生成独立副本，按下述规则清洗、划分和构造特征，不修改原始文件。

## 结果与复现

`01_data_analysis.ipynb` 用数据对比表和字段表展示形状、缺失值、重复行、数据类型及唯一值数量。
一张总览图展示评分、用户活跃度、电影热度和类型分布，并简要说明平均评分、高分比例及交互稀疏度。
图表保存为 `outputs/figures/01_data_overview.png`，预处理汇总表保存到 `outputs/tables/`。
图表使用英文标签，分析说明使用中文，数值由当前数据计算。
早期探索与诊断图仍保留在输出目录，精简后的 Notebook 不再展示或刷新这些旧图表。
当前分析使用完整数据，仅用于描述性探索；这些统计不作为训练特征直接输入模型。
未评分表示未知偏好，不能据此直接判定用户不喜欢电影。
当前探索和划分均不使用随机过程，配置中的 `random_seed` 为 `null`；模型训练时另行设置种子。

## 预处理与评估协议

Notebook 第 3–6 节执行预处理流程，重复运行可重新生成本地输入和汇总结果。

1. 清洗：检查必要字段、1–5 分整数评分、非负且可解析的 Unix 时间戳和有效用户/电影 ID，
   删除完全重复的评分行，并输出每步的删除数量。不同评分/时间的重复用户—电影对会报错，
   要求先明确重复评分策略。保留原始评分与 Unix 秒数，额外生成 UTC `rated_at`。
2. 划分：对每个用户按评分时间分组，目标比例为 80%/10%/10%。先在可行的时间组边界中
   选择最接近 80% 累计记录数的位置，再选择最接近 90% 的位置；距离相同时选较早边界。
   同时间戳不跨集合，实际比例可偏离目标。少于三个时间组的用户全部保留在训练集。
3. 标签：划分之后添加 `is_relevant = (rating >= 4)`，保留 1–3 分记录用于历史过滤。
   标签 0 表示未达到正反馈阈值；此阶段不将未评分电影自动标为负样本。
4. 特征：用户/电影 ID 映射只根据训练集拟合。保存显式评分 CSR、正反馈 CSR、训练目录的
   电影类型多标签 CSR，以及完整的正反馈训练序列。矩阵索引从 0 开始，序列电影 token
   为 `movie_index + 1`，token 0 留给 padding。当前不截断序列，也不对类别 ID 做连续归一化。
5. 验证历史为训练集，测试历史为训练集加验证集；模型参数仍只由训练集学习。
   三种模型共用训练目录作为候选全集，并排除用户当时已评分的全部电影，包括低分电影。
   留出集中评分至少 4、位于训练目录且历史未评分的电影作为可评估正例。
6. 共享评估用户需有正反馈训练历史、至少一个可评估正例、至少 10 个候选。
   冷启动正例与不满足条件的用户单独计数，主要排名指标适用于该共同可评估子集，
   不代表对全部冷启动电影或全部用户的效果。完整候选集由 `candidate_movie_ids` 动态生成。

本方案保证每个用户的训练历史早于其验证/测试历史，但不是全局日历切分：
不同用户的时间范围可能交叉。MovieLens 的时间戳表示评分提交时间，并非真实观影顺序；
训练序列内同一时间戳的电影仅按 ID 确定性排序，不能据此解释先后观影因果关系。
后续若进行负采样，采样池必须仅根据训练目录与训练历史构造，不读取留出标签。

生成的本地文件位于 `data/processed/`：三份划分 CSV、两份 ID 映射 CSV、
三个稀疏矩阵 NPZ、训练序列 JSONL、验证/测试上下文 JSONL、类型词表和 `config.json`。
配置记录划分、标签、候选与历史规则，以及源文件 SHA-256 和软件版本。
该目录包含交互级数据，已加入 `.gitignore`，组员应自行运行 Notebook 生成。
清洗、划分检查保留在处理流程中，Notebook 仅展示关键汇总；详细清洗记录、划分检查、输入规模及评估用户统计保存在 `outputs/tables/`。

在项目根目录运行检查：

```bash
python -m unittest discover -s tests -v
```

## ItemCF 基线

先运行 `01_data_analysis.ipynb` 生成预处理输入，再在项目根目录执行：

```bash
python train.py --model itemcf
python recommend.py --user-id 1 --stage test --top-k 5
```

打开 `notebooks/02_itemcf.ipynb` 查看原理、已保存的搜索与评估结果，修改 `USER_ID`
和 `TOP_K` 后即可重新生成对应用户的推荐；Notebook 不会自动重复调参或测试。
Notebook 只展示验证选参表、Precision/Recall/NDCG@10、平均命中数和电影标题/类型，默认推荐 5 部。
完整审计和命中分布属于可选分析，另行运行 `python audit_itemcf.py`，不再作为 Notebook 的前置步骤。

本实现是**基于正反馈的 ItemCF 排名基线**：用训练集中评分至少 4 分的二值矩阵计算
电影共现次数与余弦相似度，可乘以 `共现次数 / (共现次数 + shrinkage)` 进行收缩。
去掉自身相似度，每部历史电影仅保留最相似的若干邻居，再累加用户已知正反馈电影对
候选电影的相似度贡献。分数是排序值，不是预测星级。电影类型和人口属性不参与这版 ItemCF。
邻居截断按来源电影逐行执行，截断后的相似度矩阵可能不对称。

默认搜索 `n_neighbors = [20, 50, 100]` 和 `shrinkage = [0, 10, 50]` 的 9 种参数组合，
以验证集 NDCG@10 选参；同分依次比较 Recall@10、较小邻居数与较小收缩参数。
选定后仅评估该配置的测试表现，模型相似度仍只根据训练集拟合。
可通过 `--neighbors 20 50 --shrinkage 0 10` 指定搜索范围；参数变体属于同一种算法。

评估使用共享用户集合、完整未评分训练目录，不采样评估负例；报告按用户宏平均的
Precision@5/10、Recall@5/10、NDCG@5/10、平均命中数（MeanHits），
至少命中一部的用户比例（HitRate），以及推荐电影数占训练目录的覆盖率。
Precision = 命中数 / K，MeanHits = K × Precision；HitRate 衡量至少命中一部的用户占比。
未评分推荐不计为离线命中，但其真实偏好未知，因此 1 − Precision 不能解释为实际不喜欢比例。
新增 Precision 保留原有的验证 NDCG@10 选参规则，避免根据测试反馈改变选择标准。
Top-5/10 是本项目的实验设置，课程硬性要求为推荐至少 5 部未评分电影。
同推荐分数按原始电影 ID 升序排列；零支持候选仍保留为零分，不隐式引入其他推荐模型。
余弦相似度不直接解决长尾可靠性、流行度偏差或交互冷启动问题。

导出位置：

- `outputs/tables/itemcf_validation_search.csv`：全部验证参数、指标与耗时。
- `outputs/tables/itemcf_metrics.csv`：选定配置的验证与测试指标。
- `outputs/tables/itemcf_recommendation_examples.csv`：按 ID 顺序选择的前三位可评估测试用户，
  每位推荐 10 部电影，包含标题、类型、相似历史解释与测试命中标记。
- `outputs/tables/itemcf_run.json`：配置、指标、输入校验和、软件版本和运行耗时。
- `outputs/models/itemcf/`：本地相似度模型、运行记录和逐用户测试指标，已加入 Git 忽略规则。

`audit_itemcf.py` 固定使用当前模型，检查原始 `.dat` 与划分记录是否对齐、标签是否正确、
保存矩阵是否能由评分重建、真实数据抽样的相似度和邻居方向、所有评估历史/目标/候选及
独立命中计数，并导出以下诊断表：

- `itemcf_integrity_checks.csv`：具体检查的违规数量，非零时审计报错。
- `itemcf_recommendation_status.csv`：留出期高分、低于阈值及无评分记录的推荐数量和比例。
- `itemcf_evaluation_scope.csv`：候选规模、可观察正例数、零分推荐及训练热度等描述性统计。
- `itemcf_hit_distribution.csv`：Top-5/10 命中数量分布。
- `itemcf_validation_reference.csv`：相同验证协议下的训练热门排序诊断参照与 ItemCF 结果。

诊断参照不占用三种项目算法的选型名额，也不用于测试调参。审计通过能排除所列具体
数据或计分不一致，不能证明当前 ItemCF 变体最优。高比例无评分记录反映离线反馈稀疏，
不能据此推断这些未知推荐一定会被用户喜欢；仅在有评分推荐上计算的高分比例存在选择偏差。

推荐入口会检查预处理文件校验和；重新划分数据后，需要重新训练与评估以匹配新输入。
该算法的拟合、排名和搜索顺序均确定，不使用随机采样，因此随机种子记为 `null`，
早停不适用于这种直接计算相似邻居的训练方式。现有测试可通过上面的 unittest 命令运行。

算法参考：Badrul Sarwar, George Karypis, Joseph Konstan, and John Riedl. 2001.
[*Item-based Collaborative Filtering Recommendation Algorithms*](https://ra.ethz.ch/cdstore/www10/papers/519/sdm2.html).
本项目采用二值正反馈、Top-N 排名及可选收缩的具体实现，实验使用本项目的统一划分协议。

## LightFM

默认使用官方 `lightfm==1.17` 的 **logistic 损失**，避免本机 Windows WARP/BPR 负采样扩展的崩溃。
训练标签只根据训练评分构造：**≥4 分为 +1、1–2 分为 -1，3 分和未评分项不参与训练**。
不采样未评分项作为负例，不读取留出标签；评估正例仍按原规则定义为评分 ≥4。
用户使用 ID 特征，电影比较“仅 ID”和“ID＋类型”两种配置；类型特征与电影身份特征拼接后按行 L1 归一化。
两种配置和不同损失均属于同一种 LightFM 算法，但训练目标不同，结果中记录实际使用的损失和标签规则。

```bash
python train.py --model lightfm --loss logistic
python recommend.py --model lightfm --user-id 1 --top-k 5
```

默认比较维度 `[32, 64]`、正则化 `[1e-6, 1e-4]`、训练轮数检查点 `[10, 20]`，共 16 个验证配置。
固定随机种子 42、单线程，继续使用验证 NDCG@10 选参，选定后评估测试集；不使用测试结果调参。
可用 `--components 32 --epochs 5 10 --alpha 1e-6 --seed 42` 缩小为 4 个验证配置，先确认完整流程。
当前保存结果来自这个小网格，并非默认 16 配置搜索的完整结果。每个检查点的指标即时写入
`lightfm_search_progress.csv`，当前最佳向量另存到 `outputs/models/lightfm/search_best/`，以保留中断前的进展。

候选、评估用户及指标与 ItemCF 完全复用。LightFM 的用户向量只由训练集学习，测试时验证历史用于过滤已评分电影，不更新用户向量。
打开 `notebooks/03_lightfm.ipynb` 查看选定配置、算法指标比较，并修改用户 ID 比较两种算法推荐的电影标题与类型。

结果位于 `outputs/tables/lightfm_metrics.csv`、`lightfm_validation_search.csv`、`lightfm_run.json`、
`lightfm_recommendation_examples.csv` 和 `comparison_metrics.csv`。
选定模型的向量及逐用户测试结果保存到 `outputs/models/lightfm/`（不纳入 Git）。导出的向量用于推理，不包含继续训练所需的优化器状态。

可选 `--loss warp` 使用正反馈与内部负采样训练。当前 Windows 包存在可能导致负电影索引的随机数转换代码，
WARP 入口检测到时会拒绝运行；logistic 不调用该采样函数，无需重编译。
若需要在本机使用 WARP，可安装 [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
中的“使用 C++ 的桌面开发”，再执行 `python repair_lightfm_windows.py`。
该脚本只修正随机数转换，隔离测试通过后才备份并替换原扩展；这条重编译路径尚未在本机验证。

算法参考：Maciej Kula. 2015. [Metadata Embeddings for User and Item Cold-start Recommendations](https://arxiv.org/abs/1507.08439)。
实现依据：[官方 LightFM 文档](https://making.lyst.com/lightfm/docs/lightfm.html)。

## LRURec（WSDM 2024）

在已激活的项目环境中额外安装 CPU 版 PyTorch：

```bash
python -m pip install -r requirements-lrurec.txt --index-url https://download.pytorch.org/whl/cpu
python train.py --model lrurec
python recommend.py --model lrurec --user-id 1 --top-k 5
```

LRURec 将按时间排列的高分电影 token 输入嵌入层，通过复数对角线性递推、残差前馈层和共享电影嵌入计算推荐分数。
仅使用训练集生成“历史 → 下一部高分电影”的标签，采用完整训练目录的交叉熵；补齐 token 不参与分类。
同一时间戳内仍按电影 ID 确定性排列，但相邻同时间戳的预测标签被忽略，避免把这种排列当作真实先后。
输入切为最长 50 的不重叠片段，每个相邻转移最多作为标签一次；推理取最近 50 部已知高分电影。

默认小配置为隐藏维度 32、两层、dropout 0.2、batch size 64、AdamW（学习率 0.001，weight decay 0.01）、
随机种子 42、CPU 两线程。默认在第 1、3 轮检查验证 loss，保存交叉熵最低的模型，选定后才评估测试集。
可先运行 `python train.py --model lrurec --lru-epochs 1` 检查完整流程；
可用以下命令比较第 1–20 轮；不根据测试分数调整轮数。

```bash
python train.py --model lrurec --max-epochs 20
```

验证 loss 与训练一样采用下一部电影的完整目录交叉熵，按有效目标数加权平均。
修改 `--max-epochs` 可增加从头训练的轮数，并在每一轮验证；也可用 `--lru-epochs` 仅指定若干检查点，两者互斥。
只统计共同评估用户的验证期高分电影，忽略冷启动及相邻同时间戳目标；每个窗口为此前历史预留约半数位置。
逐步预测时允许使用训练历史和更早的验证期高分电影，验证标签不参与反向传播，测试数据不参与选模。
每个检查点的 `train_loss`、`validation_loss` 和排名指标写入 `lrurec_validation_search.csv`，当前最低 loss 的权重即时保存。
保存的是推理权重，不含继续训练的优化器状态；验证 loss 相同时保留较早轮数。

验证与测试沿用共同用户、完整未评分候选和 Precision/Recall/NDCG@5/10；测试历史可包含验证期已知评分，
模型参数仅由训练集学习。Notebook `04_lrurec.ipynb` 展示选定参数、三算法比较与同一用户的电影标题和类型。
结果写入 `outputs/tables/lrurec_metrics.csv`、`lrurec_validation_search.csv`、`lrurec_run.json`、
`lrurec_recommendation_examples.csv`；模型与逐用户指标写入 `outputs/models/lrurec/`。
任一算法完成训练后都会刷新 `comparison_metrics.csv`，检查已有算法的共同输入校验和及评估人数是否一致。

依据 [LRURec 论文](https://arxiv.org/abs/2310.02367) 与[官方实现](https://github.com/yueqirex/LRURec)独立实现核心结构。
本项目采用并行前缀倍增计算递推；训练划分、正反馈筛选、时间戳处理、模型大小及选参规则均按项目统一协议调整。
因此这是课程实验实现，**不作为论文实验数值或运行效率的复现**。
PyTorch 安装方式参考[官方说明](https://pytorch.org/get-started/locally/)。

## 数据引用

F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.*
ACM Transactions on Interactive Intelligent Systems 5, 4, Article 19.
[DOI: 10.1145/2827872](https://doi.org/10.1145/2827872).

使用与分发条件以数据集随附的 `README` 为准；本项目不更改原始数据的使用条件。
