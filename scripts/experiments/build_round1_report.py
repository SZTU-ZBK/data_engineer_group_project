"""Package first-round results with English charts and a concise Chinese report."""

import hashlib
import json
from pathlib import Path
import shutil
import sys
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scripts import build_ppt_materials as ppt


def main():
    source = ROOT / "outputs/experiments/round1"
    out = source / "report"
    run = json.loads((source / "run.json").read_text(encoding="utf-8"))
    assert run["completed"] and run["baseline_artifacts_unchanged"]
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    for name in ("comparison.csv", "metrics.csv", "validation_search.csv", "selection.json", "run.json"):
        shutil.copyfile(source / name, out / "data" / name)
    data = pd.read_csv(source / "comparison.csv")
    search = pd.read_csv(source / "validation_search.csv")
    algorithms = ["ItemCF", "LightFM", "LRURec"]
    ppt.OUT = out
    ppt.style()
    fig = ppt.canvas("Round 1: Baseline vs Validation-Selected Settings",
                     "Test results for 5,765 shared users. One configuration per algorithm was frozen before test evaluation.",
                     "Source: data/comparison.csv. LRURec varies inference context at fixed weights trained with length 50.")
    grid = fig.add_gridspec(1, 3, left=.07, right=.96, top=.77, bottom=.28, wspace=.35)
    test = data.loc[data.split.eq("test")]
    for i, metric in enumerate(ppt.METRICS):
        ax = fig.add_subplot(grid[0, i])
        values = test.pivot(index="algorithm", columns="version", values=f"{metric}@10").loc[algorithms]
        x = np.arange(3)
        for offset, version, label, color in ((-.18, "baseline", "Baseline", "#B9CCE4"),
                                               (.18, "round1_selected", "Round 1", "#3575C5")):
            bars = ax.bar(x+offset, values[version], width=.34, label=label, color=color)
            for bar, value in zip(bars, values[version]):
                text = f"{value:.2%}" if metric != "NDCG" else f"{value:.4f}"
                label_offset = .03 if version == "baseline" else .085
                ax.text(bar.get_x()+bar.get_width()/2, value+values.to_numpy().max()*label_offset,
                        text, ha="center", fontsize=10)
        ax.set(title=f"{metric}@10", xticks=x, xticklabels=algorithms,
               ylim=(0, values.to_numpy().max()*1.25))
        if metric != "NDCG": ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
        if i == 0: ax.legend(fontsize=10)
    fig.text(.5, .15, "Compare validation and test changes: a tiny validation gain may not generalize.",
             ha="center", fontsize=16, color=ppt.MUTED)
    ppt.save(fig, "round1_test_comparison.png")

    fig = ppt.canvas("Round 1: Validation Experiments",
                     "Selection uses validation NDCG@10, then Recall@10. Baselines are included in each search.",
                     "Full unseen training catalog; 5,776 shared validation users. Each vertical axis starts at zero.")
    grid = fig.add_gridspec(1, 3, left=.07, right=.96, top=.76, bottom=.29, wspace=.35)
    for i, algorithm in enumerate(algorithms):
        ax = fig.add_subplot(grid[0, i])
        group = search.loc[search.algorithm.eq(algorithm)]
        if algorithm == "LightFM":
            x = np.arange(3)
            for offset, epoch, color in ((-.18, 5, "#B9CCE4"), (.18, 10, "#168C82")):
                values = group.loc[group.epochs.eq(epoch)].set_index("variant").loc[
                    ["id_history", "id_only", "id_static"], "NDCG@10"]
                ax.bar(x+offset, values, width=.34, color=color, label=f"Epoch {epoch}")
            ax.set_xticks(x, ["ID + history", "ID only", "ID + static"])
            ax.legend(fontsize=10)
        else:
            column = "half_life_days" if algorithm == "ItemCF" else "inference_length"
            group = group.sort_values(column)
            labels = [("No decay" if value == 0 else f"{int(value)}d") if algorithm == "ItemCF"
                      else str(int(value)) for value in group[column]]
            values = group["NDCG@10"]
            bars = ax.bar(labels, values, color=ppt.COLORS[algorithm], width=.6)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x()+bar.get_width()/2, value+values.max()*.025, f"{value:.4f}",
                        ha="center", fontsize=10)
            ax.set_xlabel("History half-life" if algorithm == "ItemCF" else "Inference context length")
        ax.set(title=algorithm, ylabel="Validation NDCG@10", ylim=(0, group["NDCG@10"].max()*1.25))
        ax.tick_params(axis="x", labelsize=10)
        ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
    fig.text(.5, .15, "LightFM: matched fitting labels and epoch grid. LRURec: context sensitivity, without retraining.",
             ha="center", fontsize=15, color=ppt.MUTED)
    ppt.save(fig, "round1_validation_search.png")

    lines = ["# 第一轮快速实验结果", "", f"总耗时约 {run['elapsed_seconds']/60:.1f} 分钟。"
             "沿用原数据划分、共同用户和完整候选目录，只报告 Precision、Recall、NDCG。", "",
             "## 实验范围", "",
             "- ItemCF：保留原相似度，比较不衰减及半衰期 1/7/30/90 天；年龄只由当时已知历史计算。",
             "- LightFM：相同拟合标签、种子 42、32 维、logistic、正则化 1e-6、5/10 轮；比较仅 ID、ID＋静态属性、原 ID＋历史属性。原组复用已完成结果。",
             "- LRURec：复用原第 50 轮权重，比较推理历史 25/50/100；训练窗口仍为 50，属于固定权重的上下文敏感性检查，不是三组重新训练。", "",
             "三种模型均先按验证 NDCG@10（同分看 Recall）选配置，再固定配置评估测试集；测试用户为 5,765 人。", "",
             "## 测试结果：原结果 → 第一轮选中结果", "",
             "| 算法 | Precision@10 | Recall@10 | NDCG@10 |", "|---|---:|---:|---:|"]
    results = {}
    for algorithm in algorithms:
        group = test.loc[test.algorithm.eq(algorithm)].set_index("version")
        old, new = group.loc["baseline"], group.loc["round1_selected"]
        results[algorithm] = (old, new)
        cells = [f"{old[f'{metric}@10']:.2%} → {new[f'{metric}@10']:.2%}" if metric != "NDCG"
                 else f"{old['NDCG@10']:.4f} → {new['NDCG@10']:.4f}" for metric in ppt.METRICS]
        lines.append("| "+" | ".join([algorithm, *cells])+" |")
    lines += ["", "## 解读", ""]
    cfg = run["configurations"]
    for algorithm in algorithms:
        old, new = results[algorithm]
        change = (new["NDCG@10"]/old["NDCG@10"]-1)*100
        direction = "上升" if change >= 0 else "下降"
        if algorithm == "LightFM":
            variant = {"id_only": "仅 ID", "id_static": "ID＋静态属性", "id_history": "原 ID＋历史属性"}[cfg[algorithm]["feature_set"]]
            lines.append(f"**LightFM**：验证选中 {variant}、第 {cfg[algorithm]['epochs']} 轮；测试 NDCG 相对{direction} {abs(change):.2f}%。"
                         "这说明特征选择值得优先检查；但仅 ID 的权重为 1，混合组的 ID 权重为 0.5，特征归一化与有效优化尺度也随之改变，不能把差异完全归因于某一个历史特征。")
        elif algorithm == "ItemCF":
            half = cfg[algorithm]["half_life_days"]
            chosen = "不衰减" if half is None else f"半衰期 {half} 天"
            lines.append(f"**ItemCF**：验证选中{chosen}；测试 NDCG 相对{direction} {abs(change):.2f}%。"
                         "本轮差异较小，需区分验证上的微小变化与可重复的泛化改善，不应将单次差异视为稳定收益。")
        else:
            lines.append(f"**LRURec**：验证选中推理窗口 {cfg[algorithm]['inference_length']}；测试 NDCG 相对{direction} {abs(change):.2f}%。"
                         "这只反映现有权重对历史窗口的敏感性，不能证明在该长度上重新训练也会得到相同结论。")
        lines.append("")
    lines += ["", "原模型、结果表和训练数据保持原值。新推理表示及逐用户结果位于仓库 `outputs/models/round1/`；"
              "完整配置、输入校验和和选模记录见 `data/run.json`。本轮为单随机种子快速实验，尚未做多种子复核。", "",
              "图片全部为英文：`figures/round1_test_comparison.png`、`figures/round1_validation_search.png`。"]
    (out / "README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    provenance = {path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in source.glob("*.json")}
    (out / "source_sha256.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    with ZipFile(out / "round1_results.zip", "w", ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob("*")):
            if path.is_file() and path.suffix != ".zip":
                archive.write(path, arcname="round1/"+path.relative_to(out).as_posix())
    print("\n".join(lines))


if __name__ == "__main__":
    main()
