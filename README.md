# MovieLens 1M 电影推荐系统

CS5481 小组项目一。当前完成第一步：原始数据读取与基础质量检查。

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
notebooks/01_data_analysis.ipynb     数据预览、检查结果与中文说明
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

数据概况表、字段统计、质量检查结果均位于 `01_data_analysis.ipynb` 中对应单元格的输出。
当前步骤不使用随机过程，因此暂无随机种子；数据划分与模型训练时再显式设置。
后续将在该 Notebook 中补充探索图表，并将可复用的预处理代码加入 Python 模块。

## 数据引用

F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.*
ACM Transactions on Interactive Intelligent Systems 5, 4, Article 19.
[DOI: 10.1145/2827872](https://doi.org/10.1145/2827872).

使用与分发条件以数据集随附的 `README` 为准；本项目不更改原始数据的使用条件。
