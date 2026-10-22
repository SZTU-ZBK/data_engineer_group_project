# MovieLens 1M 电影推荐系统

CS5481 小组项目一。当前已完成原始数据读取、基础质量检查以及数据探索与可视化。

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
src/data.py                         数据读取、表级与列级概况、基础质量检查
notebooks/01_data_analysis.ipynb     数据预览、质量检查、探索图表与中文分析
outputs/figures/                    Notebook 自动生成的 PNG 图表
outputs/tables/                     Notebook 自动生成的统计汇总 CSV
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
当前只检查和展示数据，不删除记录，也不进行编码、归一化或训练/验证/测试划分。

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
当前步骤不使用随机过程，因此暂无随机种子；数据划分与模型训练时再显式设置。
下一步确定清洗、标签与数据划分方案，并将可复用的预处理代码加入 Python 模块。

## 数据引用

F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.*
ACM Transactions on Interactive Intelligent Systems 5, 4, Article 19.
[DOI: 10.1145/2827872](https://doi.org/10.1145/2827872).

使用与分发条件以数据集随附的 `README` 为准；本项目不更改原始数据的使用条件。
