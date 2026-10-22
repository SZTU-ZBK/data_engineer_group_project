# MovieLens 1M 电影推荐系统

CS5481 小组项目一。当前已完成数据探索、预处理，以及第一种算法 ItemCF 的训练、验证调参、评估和推荐展示。

## 环境与运行

使用 Python 3.10 或更高版本，在项目根目录执行：

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
src/evaluate.py                     共享候选排名、Recall/NDCG 与目录覆盖率
src/recommend.py                    电影推荐、标题/类型展示与相似历史解释
train.py                           验证集调参、选定模型的最终测试与结果导出
recommend.py                       按用户 ID 输出推荐的命令行入口
notebooks/01_data_analysis.ipynb     数据探索、预处理、检查结果与中文分析
notebooks/02_itemcf.ipynb            ItemCF 原理、搜索结果、评估及用户推荐演示
outputs/figures/                    Notebook 自动生成的 PNG 图表
outputs/tables/                     Notebook 自动生成的统计汇总 CSV
data/processed/                    本地模型输入与配置，不纳入 Git
tests/test_preprocessing.py         时间边界、留出数据隔离和候选过滤测试
tests/test_itemcf.py                相似度手算、排名指标、推荐过滤和保存恢复测试
requirements.txt                    数据阶段的依赖
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

`01_data_analysis.ipynb` 包含数据概况、字段统计、质量检查，以及以下五项探索：

- 1–5 分的评分数量、比例和平均分。
- 用户评分数量分布、分位数及活跃用户评分集中度。
- 电影评分数量分布、热门电影 Top 10，以及电影热度集中度。
- 将多类型字段拆分后统计电影类型；同一部电影可以计入多个类型。
- 分别按全部电影和有评分电影计算用户—电影交互密度、稀疏度。

所有分析文字中的数值由当前数据动态生成。图表使用英文标签，分析说明使用中文。
运行全部单元格后，图表保存到 `outputs/figures/`，统计表保存到 `outputs/tables/`，
可用于报告或展示；这些文件均由 Notebook 重新生成，不包含原始用户评分记录。
当前分析使用完整数据，仅用于描述性探索；这些统计不作为训练特征直接输入模型。
未评分表示未知偏好，不能据此直接判定用户不喜欢电影。
当前探索和划分均不使用随机过程，配置中的 `random_seed` 为 `null`；模型训练时另行设置种子。

## 预处理与评估协议

Notebook 第 13–18 节执行完整预处理流程，重复运行可重新生成本地输入和汇总结果。

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
报告所用清洗审计、划分规模、泄漏检查、输入规模及评估用户统计保存在 `outputs/tables/`。

在项目根目录运行检查：

```bash
python -m unittest discover -s tests -v
```

## ItemCF 基线

先运行 `01_data_analysis.ipynb` 生成预处理输入，再在项目根目录执行：

```bash
python train.py --model itemcf
python recommend.py --user-id 1 --stage test --top-k 10
```

打开 `notebooks/02_itemcf.ipynb` 查看原理、已保存的搜索与评估结果，修改 `USER_ID`
和 `TOP_K` 后即可重新生成对应用户的推荐；Notebook 不会自动重复调参或测试。

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
Recall@5/10、NDCG@5/10，以及推荐电影数占训练目录的覆盖率。
Top-5/10 是本项目的实验设置，课程硬性要求为推荐至少 5 部未评分电影。
同推荐分数按原始电影 ID 升序排列；零支持候选仍保留为零分，不隐式引入其他推荐模型。
余弦相似度不直接解决长尾可靠性、流行度偏差或交互冷启动问题。

导出位置：

- `outputs/tables/itemcf_validation_search.csv`：全部验证参数、指标与耗时。
- `outputs/tables/itemcf_metrics.csv`：选定配置的验证与测试指标。
- `outputs/tables/itemcf_recommendation_examples.csv`：按 ID 顺序选择的前三位可评估测试用户，
  每位推荐 10 部电影，包含标题、类型、相似历史解释与测试命中标记。
- `outputs/tables/itemcf_run.json`：配置、指标、输入校验和、软件版本和运行耗时。
- `outputs/figures/09_itemcf_validation_search.png`：Notebook 生成的验证搜索图。
- `outputs/models/itemcf/`：本地相似度模型、运行记录和逐用户测试指标，已加入 Git 忽略规则。

推荐入口会检查预处理文件校验和；重新划分数据后，需要重新训练与评估以匹配新输入。
该算法的拟合、排名和搜索顺序均确定，不使用随机采样，因此随机种子记为 `null`，
早停不适用于这种直接计算相似邻居的训练方式。现有测试可通过上面的 unittest 命令运行。

算法参考：Badrul Sarwar, George Karypis, Joseph Konstan, and John Riedl. 2001.
[*Item-based Collaborative Filtering Recommendation Algorithms*](https://ra.ethz.ch/cdstore/www10/papers/519/sdm2.html).
本项目采用二值正反馈、Top-N 排名及可选收缩的具体实现，实验使用本项目的统一划分协议。

## 数据引用

F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.*
ACM Transactions on Interactive Intelligent Systems 5, 4, Article 19.
[DOI: 10.1145/2827872](https://doi.org/10.1145/2827872).

使用与分发条件以数据集随附的 `README` 为准；本项目不更改原始数据的使用条件。
