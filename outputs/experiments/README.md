# 实验归档

此目录保留实验过程与历史产物，PPT 交付使用仓库根目录的 `output/`。

| 路径 | 内容 |
|---|---|
| `round1/` | 第一轮验证搜索、前后比较、选模及模型采用记录 |
| `round1/pre_promotion_tables/` | 采用新配置前的本地表格副本，已忽略 Git；旧版本也可从提交历史查看 |
| `lightfm_capacity_partial.csv` | 未完成的容量实验：32/64 维各 10 轮、128 维 1 轮，共 21 个验证点 |
| `legacy_figures/` | 已不由当前 notebook 刷新的历史探索和诊断图 |

历史实验脚本位于 `scripts/experiments/`。当前有效模型记录与指标位于 `outputs/tables/`，权重位于本地 `outputs/models/`。

当前配置为 ItemCF 50 邻居＋90 天半衰期、LightFM 仅 ID＋32 维＋10 轮，以及 LRURec 训练窗口 50／推理窗口 100。
