# PPT 交付资料

本目录只包含当前采用方案的结果。直接把 `ppt_materials.zip` 发给制作 PPT 的同学即可。

| 顺序 | 内容 | 英文图片 |
|---|---|---|
| 1 | 数据概况 | `01_data_overview.png` |
| 2 | 数据处理与时间划分 | `02_preprocessing.png` |
| 3 | Precision、Recall、NDCG | `03_metrics.png` |
| 4–5 | ItemCF 原理、当前结果与时间衰减选参 | `04_itemcf_principle.png`、`05_itemcf_results.png` |
| 6–7 | LightFM 原理、当前结果与轮数选择 | `06_lightfm_principle.png`、`07_lightfm_results.png` |
| 8–9 | LRURec 原理、当前结果与推理窗口选择 | `08_lrurec_principle.png`、`09_lrurec_results.png` |
| 10 | 三算法当前测试结果对比 | `10_model_comparison.png` |

当前配置：ItemCF 为 50 个邻居、90 天半衰期；LightFM 为仅 ID、32 维、10 轮；LRURec 训练窗口 50、推理窗口 100、第 50 轮权重。

对应文字稿为数据处理说明、指标说明及三个算法的实验分析。图片全部使用英文，16:9、2560×1440；正文为中文。`data/` 保存当前评估及调参数据，主图为 Top-10，CSV 同时保留 Top-5。`manifest.json` 记录数据来源。

结论：本次实验的三个 Top-10 测试指标均为 **LRURec > ItemCF > LightFM**。
