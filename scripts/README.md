# 脚本入口

日常生成 PPT 资料：

```bash
python scripts/build_ppt_materials.py
```

生成器使用 `docs/ppt/` 的文字稿和 `outputs/tables/` 的当前结果，重新生成 `output/` 及压缩包。`output/` 是专用交付目录，每次重建会清理旧内容。

模型训练、推荐和审计入口仍为根目录的 `train.py`、`recommend.py`、`audit_itemcf.py`。历史实验专用脚本集中放在 [experiments/](experiments/README.md)。
