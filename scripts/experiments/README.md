# 历史实验脚本

| 脚本 | 用途 |
|---|---|
| `run_round1.py` | 第一轮快速验证：ItemCF 时间衰减、LightFM 特征、LRURec 固定权重推理窗口 |
| `promote_round1.py` | 将第一轮选中配置、指标和模型采用为当前默认方案 |
| `build_round1_report.py` | 将该轮报告写入 `outputs/experiments/round1/report/` |

这些脚本依赖第一轮当时的基线和实验记录，保留用于追溯，不是日常训练入口。已完成的实验目录禁止由实验运行器覆盖；需要新的独立实验时应使用新的输出目录及相应基线。

当前模型训练入口为根目录的 `train.py`，通过 `--model` 选择 `itemcf`、`lightfm` 或 `lrurec`；当前 PPT 交付用 `python scripts/build_ppt_materials.py`。
