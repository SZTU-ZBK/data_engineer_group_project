# 未完成实验归档

`lightfm_capacity_partial.csv` 保留整理工作区前的容量实验进度：32/64 维各 10 轮、128 维仅 1 轮，共 21 个验证检查点。

该实验未完成，没有最终测试评估。表中的训练/验证交叉熵及排名指标可用于诊断，不用于替代 `outputs/tables/lightfm_run.json` 和 `lightfm_metrics.csv` 代表的已完成实验。当前最终 LightFM 模型仍为 32 维、第 5 轮。

新实验运行期间，实时进度写入 `outputs/tables/lightfm_search_progress.csv`；完整训练成功后才写入最终搜索表、容量汇总和运行记录。需要比较最终结果时，以完成实验的运行记录为准。
