from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

from .models import RidgeContextModel, RidgeDynamicContextModel, target
from .reports import write_csv
from .validation import assign_game_folds, assign_game_halves, pearson_correlation


class SnapContextModel(RidgeContextModel):
    name = "ridge_snap_context"
    impute_missing_numeric = True
    numeric_features = [
        feature for feature in RidgeContextModel.numeric_features
        if feature != "time_to_throw_frames"
    ]


class SnapTimingModel(RidgeContextModel):
    name = "ridge_snap_timing"
    impute_missing_numeric = True


MODEL_LABELS = {
    "ridge_snap_context": "Snap context",
    "ridge_snap_timing": "Snap context + timing",
    "ridge_dynamic_context": "Full dynamic context",
    "hist_gradient_boosting": "Hist. gradient boosting",
}


def _pearson(xs: list[float], ys: list[float]) -> float:
    return pearson_correlation(xs, ys)


def _spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=values.__getitem__)
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and values[order[j]] == values[order[i]]:
                j += 1
            rank = (i + j - 1) / 2 + 1
            for k in order[i:j]:
                out[k] = rank
            i = j
        return out
    return _pearson(ranks(xs), ranks(ys))


def run_player_validity_audit(
    routes: list[dict[str, str]],
    comparison_predictions: list[dict[str, str]],
    output_dir: Path,
    cross_season_summaries: list[dict[str, str]] | None = None,
    folds: int = 5,
    split_seeds: int = 100,
    min_routes: tuple[int, ...] = (10, 20, 30, 40, 50),
    seed: int = 42,
) -> dict[str, list[dict[str, str]]]:
    """Audit receiver reliability and usage sensitivity using game-held-out route residuals."""
    routes = [r for r in routes if r.get("officialPosition") == "WR"]
    keys = lambda r: (r["gameId"], r["playId"], r["nflId"])
    route_by_key = {keys(r): r for r in routes}
    existing = {keys(r): r for r in comparison_predictions}
    if len(route_by_key) != len(routes):
        raise ValueError("Route table contains duplicate game/play/receiver keys")
    if set(route_by_key) != set(existing):
        raise ValueError("Algorithm OOF predictions do not match WR route table")

    fold_map = assign_game_folds(routes, n_folds=folds, seed=seed)
    pred_rows = []
    for row in routes:
        prior = existing[keys(row)]
        fold = fold_map[(str(int(row["week"])), row["gameId"])]
        if int(prior["fold"]) != fold + 1:
            raise ValueError("Existing OOF prediction folds differ from requested folds/seed")
        pred_rows.append({**row, **prior})

    # Add comparable snap-only and snap-plus-release-timing predictions.
    for fold in range(folds):
        train = [r for r in routes if fold_map[(str(int(r["week"])), r["gameId"])] != fold]
        held = [r for r in routes if fold_map[(str(int(r["week"])), r["gameId"])] == fold]
        models = [SnapContextModel(25.0, 30), SnapTimingModel(25.0, 30)]
        for model in models:
            model.fit(train)
        pred_by_key = {keys(out): out for out in pred_rows if int(out["fold"]) == fold + 1}
        for row in held:
            out = pred_by_key[keys(row)]
            for model in models:
                out[f"pred_delta_sep_{model.name}"] = f"{model.predict(row):.8f}"

    output_dir.mkdir(parents=True, exist_ok=True)
    models = ["ridge_snap_context", "ridge_snap_timing", "ridge_dynamic_context", "hist_gradient_boosting"]
    split_rows = []
    player_rows = []
    loo_rows = []
    # Seed offset matches the project's established game-half assignment at seed=42.
    seed_values = [seed + 10_000 + i * 37 for i in range(split_seeds)]
    for split_seed in seed_values:
        assignment = assign_game_halves(routes, seed=split_seed)
        for model in models:
            agg: dict[tuple[str, str], list[float]] = defaultdict(list)
            for r in pred_rows:
                half = assignment[(str(int(r["week"])), r["gameId"])]
                residual = float(r["delta_sep"]) - float(r[f"pred_delta_sep_{model}"])
                agg[(r["nflId"], half)].append(residual)
            all_players = sorted({k[0] for k in agg})
            for threshold in min_routes:
                eligible = [p for p in all_players if all(len(agg[(p, h)]) >= threshold for h in ("A", "B"))]
                a = [statistics.fmean(agg[(p, "A")]) for p in eligible]
                b = [statistics.fmean(agg[(p, "B")]) for p in eligible]
                split_rows.append({"seed": str(split_seed), "model": model, "min_routes_per_half": str(threshold), "receivers": str(len(eligible)), "pearson": f"{_pearson(a,b):.8f}", "spearman": f"{_spearman(a,b):.8f}"})
                if split_seed == seed + 10_000:
                    for p, av, bv in zip(eligible, a, b):
                        player_rows.append({"model": model, "min_routes_per_half": str(threshold), "nflId": p, "displayName": next(r["displayName"] for r in pred_rows if r["nflId"] == p), "routes_half_a": str(len(agg[(p,"A")])), "routes_half_b": str(len(agg[(p,"B")])), "soe_half_a": f"{av:.8f}", "soe_half_b": f"{bv:.8f}"})
                    if threshold == 20:
                        full = _pearson(a,b)
                        for omitted in eligible:
                            pairs = [(x,y) for p,x,y in zip(eligible,a,b) if p != omitted]
                            corr = _pearson([x for x,_ in pairs], [y for _,y in pairs])
                            loo_rows.append({"model": model, "omitted_receiver": omitted, "displayName": next(r["displayName"] for r in pred_rows if r["nflId"] == omitted), "receivers": str(len(eligible)-1), "pearson_without_player": f"{corr:.8f}", "change_vs_full": f"{corr-full:.8f}"})

    reliability_summary = []
    for model in models:
        for threshold in min_routes:
            selected = [r for r in split_rows if r["model"] == model and int(r["min_routes_per_half"]) == threshold]
            for metric in ("pearson", "spearman", "receivers"):
                vals = sorted(float(r[metric]) for r in selected)
                reliability_summary.append({"model": model, "min_routes_per_half": str(threshold), "metric": metric, "median": f"{statistics.median(vals):.8f}", "p10": f"{vals[int(.10*(len(vals)-1))]:.8f}", "p90": f"{vals[int(.90*(len(vals)-1))]:.8f}", "split_seeds": str(len(vals))})

    # Route depth is realized after the snap: use it for residual diagnostics only.
    depth_rows = []
    for model in models:
        bins: dict[str, list[tuple[float,float]]] = defaultdict(list)
        for r in pred_rows:
            depth = float(r["route_depth"])
            label = "<2 yd" if depth < 2 else "2–5 yd" if depth < 5 else "5–10 yd" if depth < 10 else "10+ yd"
            bins[label].append((float(r["delta_sep"]), float(r[f"pred_delta_sep_{model}"])))
        for label, pairs in bins.items():
            ys, ps = [x for x,_ in pairs], [y for _,y in pairs]
            residuals = [y-p for y,p in pairs]
            depth_rows.append({"model":model,"route_depth_bin":label,"routes":str(len(pairs)),"mean_target":f"{statistics.fmean(ys):.6f}","mean_prediction":f"{statistics.fmean(ps):.6f}","mean_residual":f"{statistics.fmean(residuals):.6f}","rmse":f"{math.sqrt(statistics.fmean(e*e for e in residuals)):.6f}"})
        alignments: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for r in pred_rows:
            alignments[r["pff_positionLinedUp"] or "Unknown"].append((float(r["delta_sep"]), float(r[f"pred_delta_sep_{model}"])))
        for label, pairs in alignments.items():
            ys, ps = [x for x, _ in pairs], [y for _, y in pairs]
            residuals = [y-p for y,p in pairs]
            depth_rows.append({"model":model,"route_depth_bin":f"alignment: {label}","routes":str(len(pairs)),"mean_target":f"{statistics.fmean(ys):.6f}","mean_prediction":f"{statistics.fmean(ps):.6f}","mean_residual":f"{statistics.fmean(residuals):.6f}","rmse":f"{math.sqrt(statistics.fmean(e*e for e in residuals)):.6f}"})

    usage_rows=[]
    by_player: dict[str,list[dict[str,str]]] = defaultdict(list)
    for r in pred_rows: by_player[r["nflId"]].append(r)
    for p, rs in by_player.items():
        shallow=sum(float(r["route_depth"])<2 for r in rs)/len(rs)
        usage_rows.append({"nflId":p,"displayName":rs[0]["displayName"],"routes":str(len(rs)),"shallow_route_share":f"{shallow:.8f}","median_route_depth":f"{statistics.median(float(r['route_depth']) for r in rs):.6f}","median_snap_separation":f"{statistics.median(float(r['sep_snap']) for r in rs):.6f}","alignment_mode":max((sum(x["pff_positionLinedUp"]==v for x in rs),v) for v in {x["pff_positionLinedUp"] for x in rs})[1]})

    cross_season_loo = []
    if cross_season_summaries:
        eligible = [r for r in cross_season_summaries if r.get("eligible", "").lower() == "true"]
        for model, col21, col23 in (
            ("static", "centered_residual_2021_static", "centered_residual_2023_static"),
            ("dynamic", "centered_residual_2021_dynamic", "centered_residual_2023_dynamic"),
        ):
            xs = [float(r[col21]) for r in eligible]
            ys = [float(r[col23]) for r in eligible]
            full = _pearson(xs, ys)
            for i, row in enumerate(eligible):
                reduced_x = xs[:i] + xs[i+1:]
                reduced_y = ys[:i] + ys[i+1:]
                corr = _pearson(reduced_x, reduced_y)
                cross_season_loo.append({"model":model,"nflId":row["nflId"],"displayName":row["displayName"],"receivers":str(len(eligible)-1),"pearson_without_player":f"{corr:.8f}","change_vs_full":f"{corr-full:.8f}"})
        pairs = sorted({row["nflId"] for row in eligible if row["displayName"] in {"Rondale Moore", "Isaiah McKenzie"}})
        if len(pairs) == 2:
            remaining = [r for r in eligible if r["nflId"] not in pairs]
            for model, col21, col23 in (
                ("static", "centered_residual_2021_static", "centered_residual_2023_static"),
                ("dynamic", "centered_residual_2021_dynamic", "centered_residual_2023_dynamic"),
            ):
                cross_season_loo.append({"model":model,"nflId":"combined","displayName":"Rondale Moore + Isaiah McKenzie","receivers":str(len(remaining)),"pearson_without_player":f"{_pearson([float(r[col21]) for r in remaining],[float(r[col23]) for r in remaining]):.8f}","change_vs_full":f"{_pearson([float(r[col21]) for r in remaining],[float(r[col23]) for r in remaining])-_pearson([float(r[col21]) for r in eligible],[float(r[col23]) for r in eligible]):.8f}"})

    output = {"route_oof_predictions":pred_rows,"split_half_seeds":split_rows,"reliability_summary":reliability_summary,"receiver_half_scores":player_rows,"leave_one_receiver_out":loo_rows,"cross_season_leave_one_receiver_out":cross_season_loo,"route_depth_diagnostics":depth_rows,"receiver_usage":usage_rows}
    for name, rows_out in output.items():
        columns = list(dict.fromkeys(key for row in rows_out for key in row))
        write_csv(output_dir / f"{name}.csv", columns, rows_out)
    return output
