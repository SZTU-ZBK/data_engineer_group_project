"""Promote validated configurations and refresh canonical metrics and examples."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.evaluate import write_comparison
from src.itemcf_history import add_context_weights, decay_histories
from src.models.item_cf import ItemCF
from src.models.lightfm import LightFMRecommender
from src.models.lrurec import LRURec
from src.recommend import load_movie_metadata, recommendation_table


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    tables, models = ROOT / "outputs/tables", ROOT / "outputs/models"
    experiment = ROOT / "outputs/experiments/round1"
    result = json.loads((experiment / "run.json").read_text(encoding="utf-8"))
    if not result["completed"]:
        raise ValueError("Round one did not finish.")
    archive = experiment / "pre_promotion_tables"
    model_archive = models / "archive/pre_round1"
    if not archive.exists():
        for relative, expected in result["baseline_artifact_sha256"].items():
            if sha(ROOT / relative) != expected:
                raise ValueError(f"Baseline changed before promotion: {relative}")
        archive.mkdir(parents=True)
        for path in tables.iterdir():
            if path.is_file(): shutil.copy2(path, archive / path.name)
        for slug in ("itemcf", "lightfm", "lrurec"):
            shutil.copytree(models / slug, model_archive / slug)
    comparison = pd.read_csv(experiment / "comparison.csv")
    search = pd.read_csv(experiment / "validation_search.csv")
    metric_keys = ["users", *result["metric_names"]]
    configs = result["configurations"]
    for label, slug in (("ItemCF", "itemcf"), ("LightFM", "lightfm"), ("LRURec", "lrurec")):
        run = json.loads((archive / f"{slug}_run.json").read_text(encoding="utf-8"))
        metrics = comparison.loc[comparison.algorithm.eq(label) & comparison.version.eq("round1_selected")]
        for stage in ("validation", "test"):
            row = metrics.loc[metrics.split.eq(stage)].iloc[0]
            run[f"{stage}_metrics"] = {key: int(row[key]) if key == "users" else float(row[key]) for key in metric_keys}
        run["configuration_selection"] = "Validation NDCG@10, then Recall@10; test evaluated after freezing configurations"
        active_search = search.loc[search.algorithm.eq(label)].copy()
        if slug == "itemcf":
            model = ItemCF.load(model_archive / slug)
            model.half_life_days = configs[label]["half_life_days"]
            model.save(models / slug)
            run["selected_parameters"]["half_life_days"] = model.half_life_days
            run["grid"] = {"n_neighbors": [50], "shrinkage": [0], "half_life_days": [0, 1, 7, 30, 90]}
            run["scoring"] = "Sum similarities weighted by 2**(-age_days/90); age anchored at each user's latest known rating"
            run["input_sha256"].update({name: sha(ROOT / "data/processed" / name) for name in ("train.csv", "validation.csv")})
            active_search["n_neighbors"], active_search["shrinkage"] = 50, 0
            active_search["selected"] = active_search.half_life_days.eq(model.half_life_days)
        elif slug == "lightfm":
            model = LightFMRecommender.load(models / "round1/lightfm")
            model.save(models / slug)
            np.savez_compressed(models / slug / "feature_embeddings.npz",
                                user_feature_embeddings=model.user_embeddings, user_feature_biases=model.user_biases,
                                item_feature_embeddings=model.item_embeddings, item_feature_biases=model.item_biases)
            run["selected_parameters"].update({"epochs": 10, "user_features": "id_only", "item_features": "id_only"})
            run["feature_set"] = "id_only"
            run["feature_encoding"] = "Independent user and movie ID embeddings; identity matrices with unit row mass"
            run["representation_constraint"] = "Each user/movie has an independent ID embedding; no attribute or historical-statistic features"
            run["user_feature_names"] = [f"user_id:{value}" for value in model.user_ids]
            movie_ids = pd.read_csv(ROOT / "data/processed/movie_mapping.csv").sort_values("movie_index").movie_id
            run["item_feature_names"] = [f"movie_id:{value}" for value in movie_ids]
            run["reserved_prefix_rows"] = run["feature_history_rows"]
            run["feature_history_rows"] = 0
            run["statistics_protocol"] = "No statistical features. Parameter-fitting suffix labels match the controlled feature experiment; prefix rows are unused."
            run["grid"] = {"no_components": [32], "alpha": [1e-6], "epochs": [5, 10],
                           "user_features": ["id_only"], "item_features": ["id_only"]}
            for field in ("statistic_encoding", "smoothing_prior_weight", "loss_diagnostics"):
                run.pop(field, None)
            active_search = active_search.loc[active_search.variant.eq("id_only")].copy()
            active_search["no_components"], active_search["alpha"] = 32, 1e-6
            active_search["user_features"], active_search["item_features"], active_search["loss"] = "id_only", "id_only", "logistic"
            active_search["selected"] = active_search.epochs.eq(10)
            active_search.to_csv(tables / "lightfm_search_progress.csv", index=False)
            active_search.loc[active_search.selected].to_csv(tables / "lightfm_capacity_summary.csv", index=False)
            pd.DataFrame([{"feature_set": "id_only", "split": stage, **run[f"{stage}_metrics"]}
                          for stage in ("validation", "test")]).to_csv(tables / "lightfm_feature_comparison.csv", index=False)
        else:
            model = LRURec.load(models / "round1/lrurec")
            model.save(models / slug)
            run["training_max_length"], run["inference_max_length"] = 50, 100
            run["selected_parameters"].update({"max_length": 100})
            run["epoch_selection_metric"] = run["selection_metric"]
            run["selection_metric"] = "Training epoch by validation next-item cross entropy; inference length by validation NDCG@10"
            run["history_use"] = "Last 100 known positive movie tokens; weights trained with length 50"
            run["inference_length_grid"] = [25, 50, 100]
            active_search["epochs"], active_search["training_max_length"] = 50, 50
            active_search["selected"] = active_search.inference_length.eq(100)
        active_search = active_search.drop(columns=[column for column in ("algorithm", "split", "reused")
                                                   if column in active_search])
        active_search.to_csv(tables / f"{slug}_validation_search.csv", index=False)
        if slug == "lrurec": active_search.to_csv(tables / "lrurec_context_search.csv", index=False)
        shutil.copyfile(models / "round1" / f"{slug}_test_per_user.csv", models / slug / "test_per_user.csv")
        payload = json.dumps(run, ensure_ascii=False, indent=2) + "\n"
        (tables / f"{slug}_run.json").write_text(payload, encoding="utf-8")
        (models / slug / "run.json").write_text(payload, encoding="utf-8")
        pd.DataFrame([{"algorithm": label, "split": stage, **run[f"{stage}_metrics"]}
                      for stage in ("validation", "test")]).to_csv(tables / f"{slug}_metrics.csv", index=False)
    write_comparison(tables)
    contexts = pd.read_json(ROOT / "data/processed/test_contexts.jsonl", lines=True)
    train = pd.read_csv(ROOT / "data/processed/train.csv")
    known = pd.concat([train, pd.read_csv(ROOT / "data/processed/validation.csv")])
    mapping = pd.read_csv(ROOT / "data/processed/movie_mapping.csv").sort_values("movie_index")
    weighted = decay_histories(known, dict(zip(mapping.movie_id, mapping.movie_index)), 90)
    metadata = load_movie_metadata()
    test = pd.read_csv(ROOT / "data/processed/test.csv")
    rating_map = test.set_index(["user_id", "movie_id"]).rating.to_dict()
    combined = []
    for slug, cls in (("itemcf", ItemCF), ("lightfm", LightFMRecommender), ("lrurec", LRURec)):
        model = cls.load(models / slug)
        examples = []
        for context in contexts.loc[contexts.eligible].sort_values("user_id").head(3).to_dict("records"):
            if slug == "itemcf": context = add_context_weights(context, weighted)
            rec = recommendation_table(model, context, mapping.movie_id.to_numpy(), metadata, 10)
            assert not set(rec.movie_id) & set(context["seen_movie_ids"])
            rec.insert(0, "user_id", context["user_id"])
            rec["relevant_in_test"] = rec.movie_id.isin(context["relevant_movie_ids"])
            examples.append(rec)
            package = rec.copy()
            package.insert(0, "algorithm", slug)
            package["test_rating"] = [rating_map.get((context["user_id"], movie), np.nan) for movie in rec.movie_id]
            package["seen_in_history"] = False
            combined.append(package)
        pd.concat(examples, ignore_index=True).to_csv(tables / f"{slug}_recommendation_examples.csv", index=False)
    pd.concat(combined, ignore_index=True).to_csv(tables / "three_model_recommendation_comparison.csv", index=False)
    (experiment / "promotion.json").write_text(json.dumps({"promoted": True,
        "configurations": configs, "archive": archive.relative_to(ROOT).as_posix()}, indent=2), encoding="utf-8")
    print("Current models, run records, metrics and recommendations now use the selected configurations.")


if __name__ == "__main__":
    main()
