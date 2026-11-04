"""Build slide-ready figures and small source tables from completed experiments."""

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import column_summary, dataset_summary, load_movielens, resolve_data_dir
TABLES = ROOT / "outputs" / "tables"
OUT = ROOT / "output"
DOCS = ROOT / "docs/ppt"
METRICS = ("Precision", "Recall", "NDCG")
COLORS = {"ItemCF": "#3575C5", "LightFM": "#168C82", "LRURec": "#D78128"}
NAVY, MUTED = "#18314F", "#53647A"


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.unicode_minus": False,
                         "text.color": NAVY, "axes.labelcolor": NAVY,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "savefig.facecolor": "white"})


def canvas(title, subtitle, source):
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    fig.text(.05, .925, title, fontsize=27, weight="bold")
    fig.text(.05, .875, subtitle, fontsize=14, color=MUTED)
    fig.text(.05, .035, source, fontsize=10, color=MUTED)
    return fig


def save(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label in fig.findobj(match=matplotlib.text.Text):
        if re.search(r"[\u3400-\u9fff]", label.get_text()):
            raise ValueError(f"Figure text must be English: {name}: {label.get_text()}")
    fixed_labels = [*fig.texts, *[label for ax in fig.axes for label in
                                  [*ax.texts, ax.title, ax.xaxis.label, ax.yaxis.label]]]
    for label in fixed_labels:
        if label.get_visible() and label.get_text():
            bounds = label.get_window_extent(renderer)
            if (bounds.x0 < -1 or bounds.y0 < -1 or bounds.x1 > fig.bbox.width + 1
                    or bounds.y1 > fig.bbox.height + 1):
                raise ValueError(f"Figure text exceeds canvas: {name}: {label.get_text()}")
    fig.savefig(OUT / "figures" / name, dpi=160)
    plt.close(fig)


def box(ax, x, y, width, height, title, body, color="#3575C5"):
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.012",
                              facecolor="#F2F6FA", edgecolor=color, linewidth=1.8))
    ax.text(x + width / 2, y + height - .055, title, ha="center", va="top",
            fontsize=17, weight="bold", color=color)
    ax.text(x + width / 2, y + height / 2 - .025, body, ha="center", va="center",
            fontsize=14, linespacing=1.7)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=20,
                                color=MUTED, linewidth=1.8))


def principle(name, title, subtitle, blocks, equation, note):
    fig = canvas(title, subtitle, "Source: project implementation and completed run records. Architecture shown as a simplified schematic.")
    ax = fig.add_axes([0, .1, 1, .71]); ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    xs = [.055, .295, .535, .775]
    for i, (heading, body) in enumerate(blocks):
        box(ax, xs[i], .38, .17, .47, heading, body, COLORS[name])
        if i < 3:
            arrow(ax, (xs[i] + .185, .615), (xs[i + 1] - .02, .615))
    ax.text(.5, .22, equation, fontsize=20, ha="center", weight="bold")
    ax.text(.5, .07, note, fontsize=14, ha="center", color=MUTED)
    save(fig, {"ItemCF": "04_itemcf_principle.png", "LightFM": "06_lightfm_principle.png",
               "LRURec": "08_lrurec_principle.png"}[name])


def bar_metrics(fig, grid, model, metrics):
    rows = metrics.loc[metrics.algorithm.eq(model)].set_index("split")
    for i, metric in enumerate(METRICS):
        ax = fig.add_subplot(grid[0, i])
        values = [rows.loc[split, f"{metric}@10"] for split in ("validation", "test")]
        bars = ax.bar(["Validation", "Test"], values, color=["#B9CCE4", COLORS[model]], width=.55)
        ax.set(title=f"{metric}@10", ylim=(0, max(values) * 1.35))
        ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
        if metric != "NDCG": ax.yaxis.set_major_formatter(PercentFormatter(1))
        for bar, value in zip(bars, values):
            label = f"{value:.2%}" if metric != "NDCG" else f"{value:.4f}"
            ax.text(bar.get_x() + bar.get_width()/2, value + max(values)*.06,
                    label, ha="center", weight="bold", fontsize=15)


def build():
    style()
    documents = ["README.md", "数据处理说明.md", "评估指标说明.md", "ItemCF实验分析.md", "LightFM实验分析.md", "LRURec实验分析.md", "PPT讲解要点.md"]
    if not all((DOCS / name).is_file() for name in documents):
        raise FileNotFoundError("PPT document templates are missing.")
    if OUT.resolve() != ROOT.resolve() / "output":
        raise ValueError("Refusing to clean an output directory outside the expected workspace path.")
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "data").mkdir(exist_ok=True)
    for name in documents:
        shutil.copyfile(DOCS / name, OUT / name)
    metric_cols = [f"{metric}@{k}" for k in (5, 10) for metric in METRICS]
    metrics = pd.read_csv(TABLES / "comparison_metrics.csv")
    metrics = metrics[["algorithm", "split", "users", *metric_cols]]
    assert len(metrics) == 6 and set(metrics.algorithm) == set(COLORS)
    sources = []

    def export(name, target=None, columns=None):
        path = TABLES / name
        table = pd.read_csv(path)
        if columns is not None: table = table[columns]
        table.to_csv(OUT / "data" / (target or name), index=False, encoding="utf-8-sig")
        sources.append(path)
        return table

    export("comparison_metrics.csv", "evaluation_metrics.csv", list(metrics.columns))
    search_fields = {"ItemCF": ["half_life_days", "n_neighbors", "shrinkage", "users", *metric_cols, "selected"],
                     "LightFM": ["epochs", "no_components", "alpha", "users", *metric_cols, "selected"],
                     "LRURec": ["inference_length", "training_max_length", "epochs", "users", *metric_cols, "selected"]}
    searches = {model: export("lrurec_context_search.csv" if model == "LRURec"
                              else f"{model.lower()}_validation_search.csv", columns=search_fields[model])
                for model in COLORS}
    summary = export("eda_summary.csv").set_index("metric")["value"]
    ratings = export("rating_distribution.csv")
    genres = export("genre_distribution.csv")
    splits = export("split_summary.csv")
    export("cleaning_audit.csv"); export("evaluation_cohorts.csv"); export("model_input_summary.csv")
    recommendations = pd.read_csv(TABLES / "three_model_recommendation_comparison.csv")
    recommendations.loc[recommendations.user_id.eq(1) & recommendations["rank"].le(5),
                        ["algorithm", "rank", "title", "genres", "test_rating", "relevant_in_test"]].to_csv(
        OUT / "data" / "recommendations_user1.csv", index=False, encoding="utf-8-sig")
    sources.append(TABLES / "three_model_recommendation_comparison.csv")
    runs = {}
    for model in COLORS:
        path = TABLES / f"{model.lower()}_run.json"; sources.append(path)
        run = json.loads(path.read_text(encoding="utf-8")); runs[model] = run
        for split in ("validation", "test"):
            row = metrics.loc[metrics.algorithm.eq(model) & metrics.split.eq(split)].iloc[0]
            assert row.users == run[f"{split}_metrics"]["users"]
            for column in metric_cols:
                assert abs(row[column] - run[f"{split}_metrics"][column]) < 1e-12
    for key in ("config.json", "movie_mapping.csv", "test_contexts.jsonl"):
        assert len({run["input_sha256"][key] for run in runs.values()}) == 1
    settings = {model: {"selected_parameters": run["selected_parameters"],
                        "selection_metric": run["selection_metric"],
                        "training_feedback": run.get("training_feedback", run.get("input", run.get("loss"))),
                        "random_seed": run["random_seed"]} for model, run in runs.items()}
    for model, run in runs.items():
        settings[model].update({key: run[key] for key in ("training_max_length", "inference_max_length") if key in run})
    (OUT / "data" / "model_settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

    fig = canvas("MovieLens 1M: Dataset Overview", "Most observed ratings are 3–5 stars. Ratings of 4–5 define positive feedback.", "Source: eda_summary.csv / rating_distribution.csv / genre_distribution.csv. Full-data statistics are descriptive only.")
    cards = [("Users", f"{int(summary['users']):,}"), ("Catalog movies", f"{int(summary['catalog_movies']):,}"),
             ("Observed ratings", f"{int(summary['ratings']):,}"), ("Share of 4–5 ratings", f"{summary['rating_4_or_5_share']:.2%}")]
    for i, (label, value) in enumerate(cards):
        fig.text(.075+i*.235, .755, value, fontsize=29, weight="bold", color="#3575C5")
        fig.text(.075+i*.235, .705, label, fontsize=15, color=MUTED)
    ax = fig.add_axes([.08,.18,.38,.40]); bars=ax.bar(ratings.rating.astype(str), ratings["count"],
        color=["#BDCCDC"]*3+["#168C82"]*2)
    ax.set(title="Rating distribution", xlabel="Rating", ylabel="Number of ratings")
    for b,v in zip(bars,ratings["count"]): ax.text(b.get_x()+b.get_width()/2,v+5000,f"{v:,}",ha="center",fontsize=11)
    ax.set_ylim(0,ratings["count"].max()*1.2)
    ax=fig.add_axes([.59,.18,.34,.40]); top=genres.head(6).iloc[::-1]
    ax.barh(top.genre,top.movie_count,color="#3575C5"); ax.set(title="Top six movie genres",xlabel="Number of movies")
    fig.text(.52,.09,"A movie may have multiple genres; genre counts are not mutually exclusive.",ha="center",color=MUTED,fontsize=12)
    save(fig,"01_data_overview.png")

    fig=canvas("Preprocessing: Split Before Building Features", "No rows were removed. Equal-timestamp events of a user stay in the same partition.", "Source: cleaning_audit.csv / split_summary.csv / evaluation_cohorts.csv. Model parameters are fitted on training data only.")
    ax=fig.add_axes([0,.47,1,.34]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    blocks=[("Quality checks","Missing values / IDs\nRating range / duplicates\n0 rows removed"),("Temporal split","Train / validation / test\nAbout 80% / 10% / 10%\nPreserve timestamp groups"),
            ("Training inputs","Positive: rating ≥4\n6,040 × 3,667 matrix\nGenres / positive sequences"),("Shared evaluation","Exclude all rated items\nValidation: 5,776 users\nTest: 5,765 users")]
    for i,(title,body) in enumerate(blocks):
        box(ax,.055+i*.24,.06,.17,.87,title,body)
        if i<3:arrow(ax,(.24+i*.24,.49),(.275+i*.24,.49))
    ax=fig.add_axes([.10,.25,.80,.10]);left=0
    for (_,row),label,color in zip(splits.iterrows(),["Train","Validation","Test"],["#3575C5","#168C82","#D78128"]):
        share=row.row_share;ax.barh([0],share,left=left,color=color,height=.65)
        ax.text(left+share/2,0,f"{label}\n{int(row.rows):,}\n{share:.2%}",color="white",ha="center",va="center",fontsize=12)
        left+=share
    ax.set_xlim(0,1);ax.axis("off")
    fig.text(.5,.14,"Validation history: train. Test history: train + validation. Rank the full unseen training catalog.",ha="center",fontsize=14)
    save(fig,"02_preprocessing.png")

    fig=canvas("Evaluation: Precision, Recall and NDCG", "Example: recommend A, B, C, D, E at Top-5. Relevant targets are A, C, F.", "Binary relevance: rating ≥4. Metrics are macro-averaged over users. Unrated items are unknown, not confirmed dislikes.")
    ax=fig.add_axes([.04,.49,.92,.30]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    for i,letter in enumerate("ABCDE"):
        color="#168C82" if letter in "AC" else "#53647A"
        box(ax,.035+i*.195,.13,.14,.68,f"Rank {i+1}",letter+("  Hit" if letter in "AC" else ""),color)
    ndcg=1.5/(1+1/np.log2(3)+.5)
    for x,title,value,detail in zip([.20,.50,.80],METRICS,["40%","66.67%",f"{ndcg:.3f}"],
                                  ["2 / 5","2 / 3","(1 + 0.5) / ideal DCG"]):
        fig.text(x,.41,title+"@5",ha="center",fontsize=22,weight="bold")
        fig.text(x,.30,value,ha="center",fontsize=34,color="#3575C5",weight="bold")
        fig.text(x,.22,detail,ha="center",fontsize=14,color=MUTED)
    fig.text(.5,.11,"Main result charts use K=10. Source tables also include K=5. Higher is better for all three metrics.",ha="center",fontsize=15)
    save(fig,"03_metrics.png")

    principle("ItemCF","ItemCF: Similarity with Time-Weighted History","Selected configuration: 50 neighbors, shrinkage = 0, history half-life = 90 days.",
        [("Positive feedback","Train ratings ≥4\nUser–item binary matrix\nObserved positives only"),("Item similarity","Count co-occurrences\nCosine normalization\nRemove self-similarity"),
         ("Weighted history","Top-50 per source item\nWeight: 2 ** (−age / 90)\nAge from known history"),("Recommendation","Exclude all rated items\nSort candidate scores\nReturn Top-K")],
        r"$s(u,j) = \sum_{i \in H_u^+} 2^{-age_i/90}\,\widetilde{sim}(i,j)$","Select history half-life by validation NDCG@10. Age is measured in days; scores are not star ratings.")
    principle("LightFM","LightFM: Identity Embeddings for Collaborative Ranking","Current configuration: ID-only features, 32 dimensions, logistic loss, L2 = 1e-6, epoch 10.",
        [("Fitting labels","Use fixed training suffix\nPositive / negative ratings\nNo unrated-item labels"),("Identity inputs","6,040 user IDs\n3,667 movie IDs\nNo attribute features"),
         ("Embeddings","One vector per ID\nUnit identity weight\n32-dimensional vectors"),("Ranking","User–item dot product\nAdd learned biases\nFilter history, take Top-K")],
        r"$s(u,i) = p_u^\top q_i + b_u + b_i$","Labels: ratings ≥4 are +1; ratings 1–2 are −1. Rating 3 and unrated items are omitted.")
    principle("LRURec","LRURec: Next-Item Prediction from Positive Sequences","Current configuration: hidden size 32, two blocks, training length 50, inference context 100.",
        [("Positive sequence","Sort by rating timestamp\nKeep ratings ≥4\nMask adjacent time ties"),("Item embeddings","Training windows: 50\nInference context: 100\nPad shorter histories"),
         ("Recurrent state","Complex diagonal LRU\nResidual + feedforward\nAccumulate history"),("Next-item ranking","Use last 100 known items\nExclude all rated items\nReturn Top-K")],
        r"$h_t = \Lambda h_{t-1} + Bx_t$","Epoch selected by validation cross entropy; inference context selected by validation NDCG@10.")

    for model in COLORS:
        subtitles={"ItemCF":"Cosine neighbors with a 90-day history half-life, selected on validation NDCG@10.",
                   "LightFM":"ID-only features. Epoch 10 has higher validation Precision and NDCG, with slightly lower Recall.",
                   "LRURec":"Best test results on all three metrics. Weights trained at length 50; inference uses length 100."}
        fig=canvas(model+": Experimental Results",subtitles[model],"Shared evaluation: 5,776 validation users; 5,765 test users. Full unseen training catalog; macro-averaged metrics.")
        grid=fig.add_gridspec(2,3,left=.08,right=.95,top=.79,bottom=.18,hspace=.65)
        bar_metrics(fig,grid,model,metrics)
        search=searches[model]
        if model=="ItemCF":
            ax=fig.add_subplot(grid[1,:])
            for metric,color in zip(METRICS,["#3575C5","#168C82","#D78128"]):
                ax.plot(range(len(search)), search[f"{metric}@10"], marker="o", color=color, label=f"{metric}@10")
            ax.set(xticks=range(len(search)), xticklabels=["None" if value == 0 else f"{value:g}d" for value in search.half_life_days],
                   xlabel="Positive-history half-life", ylabel="Validation score", title="Validation Metrics by History Half-life", ylim=(0,None))
            ax.grid(alpha=.15); ax.legend(ncol=3, fontsize=11)
        else:
            ax=fig.add_subplot(grid[1,:])
            x_column = "epochs" if model == "LightFM" else "inference_length"
            search = search.sort_values(x_column)
            for metric,color in zip(METRICS,["#3575C5","#168C82","#D78128"]):
                ax.plot(search[x_column],search[f"{metric}@10"],label=f"{metric}@10",color=color,
                        marker="o",linewidth=2)
            selected=int(runs[model]["selected_parameters"]["epochs" if model == "LightFM" else "max_length"])
            ax.axvline(selected,color=MUTED,linestyle="--",alpha=.7)
            ax.set(xlabel="Epoch" if model == "LightFM" else "Inference context length",ylabel="Validation score",
                   title="Validation Metrics by Epoch" if model == "LightFM" else "Validation Metrics by Inference Context",ylim=(0,None))
            ax.set_xticks(search[x_column])
            ax.grid(alpha=.15);ax.legend(loc="best",ncol=3,fontsize=11)
        note={"ItemCF":"The 30-day and 90-day validation results are very close; the selected half-life is 90 days.",
              "LightFM":"The current model uses only user/movie identities; the fitting labels and parameter grid are fixed.",
              "LRURec":"Training window: 50. Inference window: 100. Context selection keeps learned weights fixed."}[model]
        fig.text(.5,.095,note,ha="center",fontsize=13,color=MUTED)
        save(fig,{"ItemCF":"05_itemcf_results.png","LightFM":"07_lightfm_results.png","LRURec":"09_lrurec_results.png"}[model])

    fig=canvas("Model Comparison: LRURec > ItemCF > LightFM","Test Top-10 results for 5,765 shared users, under the current experimental configurations.", "Source: comparison_metrics.csv. Training objectives, fitting samples and model-selection rules differ across models.")
    test=metrics.loc[metrics.split.eq("test")].set_index("algorithm").loc[list(COLORS)]
    grid=fig.add_gridspec(1,3,left=.07,right=.96,top=.76,bottom=.29,wspace=.35)
    for i,metric in enumerate(METRICS):
        ax=fig.add_subplot(grid[0,i]);values=test[f"{metric}@10"]
        bars=ax.bar(list(COLORS),values,color=list(COLORS.values()),width=.60)
        ax.set(title=f"{metric}@10",ylim=(0,values.max()*1.28));ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True)
        if metric!="NDCG":ax.yaxis.set_major_formatter(PercentFormatter(1))
        for bar,value in zip(bars,values):
            ax.text(bar.get_x()+bar.get_width()/2,value+values.max()*.04,
                    f"{value:.2%}" if metric!="NDCG" else f"{value:.4f}",ha="center",weight="bold",fontsize=14)
    gain_p=test.loc['LRURec','Precision@10']/test.loc['ItemCF','Precision@10']-1
    gain_n=test.loc['LRURec','NDCG@10']/test.loc['ItemCF','NDCG@10']-1
    fig.text(.5,.17,f"LRURec vs ItemCF: +{gain_p:.2%} Precision@10 and +{gain_n:.2%} NDCG@10 (relative gains).",ha="center",fontsize=18,weight="bold")
    fig.text(.5,.10,"Offline results under this protocol. Unrated preferences are unknown; multi-seed stability remains untested.",ha="center",fontsize=14,color=MUTED)
    save(fig,"10_model_comparison.png")

    raw = load_movielens()
    raw_info = dataset_summary(raw).reset_index()
    fields = column_summary(raw).reset_index()
    raw_info.to_csv(OUT / "data/dataset_info.csv", index=False, encoding="utf-8-sig")
    fields.to_csv(OUT / "data/column_info.csv", index=False, encoding="utf-8-sig")
    config = json.loads((ROOT / "data/processed/config.json").read_text(encoding="utf-8"))
    for table in raw:
        path = resolve_data_dir() / f"{table}.dat"
        if hashlib.sha256(path.read_bytes()).hexdigest() != config["source_sha256"][path.name]:
            raise ValueError(f"Raw dataset differs from preprocessing: {path.name}")
        sources.append(path)
    fig = canvas("Dataset Tables and Field Quality", "Raw table shapes, field types, missing counts and unique values before preprocessing.",
                 "Source: users.dat / movies.dat / ratings.dat. Aggregate statistics only; raw records are not included in this package.")
    for i, name in enumerate(("users", "movies", "ratings")):
        ax = fig.add_axes([.045 + i*.32, .24, .29, .53]); ax.axis("off")
        info = raw_info.loc[raw_info.table.eq(name)].iloc[0]
        ax.set_title(f"{name.title()}\n{int(info.rows):,} rows × {int(info['columns'])} columns", fontsize=17, pad=18)
        group = fields.loc[fields.table.eq(name)]
        cells = [[row.column, row.dtype, str(row.missing_values), f"{row.unique_values:,}"] for row in group.itertuples()]
        table = ax.table(cellText=cells, colLabels=["Column", "Type", "Nulls", "Unique"],
                         colWidths=[.37,.20,.16,.27], cellLoc="left", bbox=[0,.10,1,.75])
        table.auto_set_font_size(False); table.set_fontsize(11)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#D6E0EB")
            if row == 0: cell.set_facecolor(NAVY); cell.set_text_props(color="white", weight="bold")
            else: cell.set_facecolor("#F2F6FA" if row % 2 else "white")
        ax.text(.5,-.08,f"Missing values: {int(info.missing_values):,}\nExact duplicate rows: {int(info.duplicate_rows):,}",
                ha="center",va="top",fontsize=13,color=MUTED)
    combinations = int(fields.loc[fields.table.eq("movies") & fields.column.eq("genres"), "unique_values"].iloc[0])
    fig.text(.5,.10,f"Genres: {combinations} combinations of {len(genres)} individual labels. Age and occupation are category codes.\nTimestamps record rating submissions; unique genre strings are not individual genre labels.",
             ha="center",fontsize=13,color=MUTED)
    save(fig,"11_dataset_info.png")

    example = recommendations.loc[recommendations.user_id.eq(1) & recommendations["rank"].le(5)].copy()
    order = {"itemcf": 0, "lightfm": 1, "lrurec": 2}
    example["model_order"] = example.algorithm.map(order)
    example = example.sort_values(["model_order", "rank"])
    fig = canvas("Recommendation Example: User 1", "Each algorithm recommends five movies outside the user's known rating history.",
                 "Source: recommendations_user1.csv. Unrated test items are unknown preferences; they are not confirmed dislikes.")
    ax = fig.add_axes([.04,.15,.92,.66]);ax.axis("off")
    cells=[]
    for row in example.itertuples():
        label = "Unrated" if pd.isna(row.test_rating) else f"{row.test_rating:g} stars"
        if row.relevant_in_test: label += " (hit)"
        cells.append([{"itemcf":"ItemCF","lightfm":"LightFM","lrurec":"LRURec"}[row.algorithm],
                      str(row.rank),row.title,row.genres,label])
    table = ax.table(cellText=cells, colLabels=["Model","Rank","Movie title","Genres","Test feedback"],
                     colWidths=[.09,.045,.445,.30,.12],cellLoc="left",bbox=[0,0,1,1])
    table.auto_set_font_size(False);table.set_fontsize(9.7)
    for (row, col),cell in table.get_celld().items():
        cell.set_edgecolor("#D6E0EB")
        if row==0:cell.set_facecolor(NAVY);cell.set_text_props(color="white",weight="bold")
        else:
            cell.set_facecolor("#F2F6FA" if ((row-1)//5)%2==0 else "white")
            if col==0:cell.set_text_props(color=COLORS[cells[row-1][0]],weight="bold")
    fig.text(.5,.095,"One example illustrates recommendation behavior; aggregate quality is evaluated over all 5,765 test users.",
             ha="center",fontsize=13,color=MUTED)
    save(fig,"12_recommendation_examples.png")

    sources += [ROOT / path for path in ["src/data.py", "src/evaluate.py", "src/itemcf_history.py", "src/models/item_cf.py", "src/models/lightfm.py", "src/models/lrurec.py", "scripts/build_ppt_materials.py"]]
    sources += [DOCS / name for name in documents]
    manifest={"snapshot_date":"2026-11-04","dataset":"MovieLens 1M","metrics":list(METRICS),
              "k_values":[5,10],"main_figures_k":10,"figure_language":"en","analysis_language":"zh-CN",
              "figures":sorted(p.name for p in (OUT/'figures').glob('*.png')),
              "sources":{path.relative_to(ROOT).as_posix():hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with ZipFile(OUT/'ppt_materials.zip','w',compression=ZIP_DEFLATED) as archive:
        for path in sorted(OUT.rglob('*')):
            if path.is_file() and path.suffix != '.zip':
                archive.write(path,arcname='ppt_materials/'+path.relative_to(OUT).as_posix())
    print(f"Saved {len(manifest['figures'])} figures and source tables to {OUT}")


if __name__=="__main__":
    build()
