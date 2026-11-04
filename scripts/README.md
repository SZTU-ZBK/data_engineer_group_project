# 脚本入口

日常生成 PPT 资料：

```bash
python scripts/build_ppt_materials.py
```

生成器使用 `docs/ppt/` 的文字稿和 `outputs/tables/` 的当前结果，重新生成 `output/` 及压缩包。`output/` 是专用交付目录，每次重建会清理旧内容。

交付包含 12 张英文图，以及数据处理、指标、三算法分析和 5 分钟讲解提纲。生成数据质量页还需本地 MovieLens 三份原始 `.dat` 文件及 `data/processed/config.json`；生成器校验数据来源，只导出汇总统计，不打包原始评分记录。

模型训练、推荐和审计入口仍为根目录的 `train.py`、`recommend.py`、`audit_itemcf.py`。历史实验专用脚本集中放在 [experiments/](experiments/README.md)。
