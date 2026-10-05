# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Game-Grouped Out-of-Fold Receiver Reliability
#
# The earlier receiver split-half diagnostic used only a small validation subset and had too few qualifying receivers to say much. This experiment generates a prediction for every WR route from a model that did not train on that route's game. Five folds are assigned within each week, and static and dynamic ridge models use identical folds.
#
# To test whether route residuals translate into stable player evaluations, the out-of-fold routes are also divided into two disjoint, week-balanced sets of games. We correlate each qualifying receiver's mean SOE between halves. A receiver must have at least 20 WR routes in each half. Bootstrap intervals resample receivers and describe uncertainty in this dataset; they do not establish season-to-season stability.

# %%
from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from separation_over_expected.reports import read_csv_rows

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "dynamic_features" / "cross_validation"
predictions = read_csv_rows(OUTPUT_DIR / "route_level_oof_predictions_wr.csv")
metrics = read_csv_rows(OUTPUT_DIR / "oof_metrics_wr.csv")
player_halves = read_csv_rows(OUTPUT_DIR / "receiver_oof_half_scores_wr.csv")
reliability = read_csv_rows(OUTPUT_DIR / "receiver_oof_reliability_wr.csv")
len(predictions), len(metrics), len(player_halves), reliability

# %% [markdown]
# ## Verify out-of-game coverage
#
# Every WR route should appear exactly once, every game should map to one fold, and both folds and reliability halves should cover all weeks. This confirms the unit kept together during model splitting was the game, not the route.

# %%
from collections import Counter, defaultdict

route_keys = [(r["gameId"], r["playId"], r["nflId"]) for r in predictions]
folds_by_game = defaultdict(set)
halves_by_game = defaultdict(set)
for row in predictions:
    folds_by_game[(row["week"], row["gameId"])].add(row["fold"])
    halves_by_game[(row["week"], row["gameId"])].add(row["reliability_half"])
integrity = {
    "oof_routes": len(predictions),
    "unique_route_keys": len(set(route_keys)),
    "games": len(folds_by_game),
    "games_in_multiple_folds": sum(len(value) != 1 for value in folds_by_game.values()),
    "games_in_multiple_reliability_halves": sum(len(value) != 1 for value in halves_by_game.values()),
    "routes_by_fold": dict(sorted(Counter(r["fold"] for r in predictions).items())),
    "routes_by_reliability_half": dict(sorted(Counter(r["reliability_half"] for r in predictions).items())),
    "games_by_reliability_half": {half: len({r["gameId"] for r in predictions if r["reliability_half"] == half}) for half in ("A", "B")},
    "weeks": sorted({int(r["week"]) for r in predictions}),
}
integrity

# %% [markdown]
# ## Out-of-fold route prediction
#
# Each row below is scored only by a model trained on the other four folds. The overall row aggregates all held-out predictions. Fold-to-fold variation shows how sensitive route prediction is to which games are held out.

# %%
fold_metrics = [r for r in metrics if r["fold"] != "OOF_ALL"]
overall_metrics = [r for r in metrics if r["fold"] == "OOF_ALL"]
fold_metrics, overall_metrics

# %% [markdown]
# ## Receiver SOE stability
#
# Receiver SOE is mean route residual within each game half. Positive SOE means the receiver's routes produced more snap-to-release separation change than the model expected, averaged over those routes. Pearson correlation describes agreement in levels; Spearman describes agreement in ranking. The paired bootstrap difference compares dynamic and static Pearson stability on the same receivers.

# %%
reliability

# %%
player_comparison = []
for row in player_halves:
    player_comparison.append({
        "receiver": row["displayName"],
        "routes_A": row["routes_ridge_dynamic_context_A"],
        "routes_B": row["routes_ridge_dynamic_context_B"],
        "static_A": row["mean_soe_ridge_context_A"],
        "static_B": row["mean_soe_ridge_context_B"],
        "dynamic_A": row["mean_soe_ridge_dynamic_context_A"],
        "dynamic_B": row["mean_soe_ridge_dynamic_context_B"],
    })
player_comparison

# %%
class SVG:
    def __init__(self, markup):
        self.markup = markup
    def _repr_svg_(self):
        return self.markup

values = [float(row[f"mean_soe_{model}_{half}"]) for row in player_halves for model in ("ridge_context", "ridge_dynamic_context") for half in ("A", "B")]
plot_min, plot_max = min(values) - 0.2, max(values) + 0.2
def point_xy(value_x, value_y, left, top, size=330):
    scale = size - 55
    return left + 42 + (value_x - plot_min) / (plot_max - plot_min) * scale, top + 15 + scale - (value_y - plot_min) / (plot_max - plot_min) * scale

panels = []
for panel, model, color in ((0, "ridge_context", "#1769aa"), (1, "ridge_dynamic_context", "#d05b22")):
    left, top = 20 + panel * 390, 40
    size = 330
    dots = []
    for row in player_halves:
        x, y = point_xy(float(row[f"mean_soe_{model}_A"]), float(row[f"mean_soe_{model}_B"]), left, top, size)
        dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}" fill-opacity="0.68"/>')
    x0, y0 = point_xy(plot_min, plot_min, left, top, size)
    x1, y1 = point_xy(plot_max, plot_max, left, top, size)
    title = "Static ridge" if panel == 0 else "Dynamic ridge"
    panels.append(f'<text x="{left+160}" y="25" text-anchor="middle" font-size="15">{title}</text><line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="#888" stroke-dasharray="5,5"/><rect x="{left+42}" y="{top+15}" width="{size-55}" height="{size-55}" fill="none" stroke="#333"/>{"".join(dots)}<text x="{left+175}" y="{top+size+8}" text-anchor="middle" font-size="12">Mean SOE, game set A</text><text x="{left+8}" y="{top+160}" transform="rotate(-90 {left+8} {top+160})" text-anchor="middle" font-size="12">Mean SOE, game set B</text>')
reliability_scatter = SVG(f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="420" viewBox="0 0 800 420"><rect width="100%" height="100%" fill="white"/>{"".join(panels)}</svg>')
reliability_scatter

# %% [markdown]
# ## Interpretation
#
# Across 20,415 WR routes, dynamic context modestly improves out-of-fold route prediction (RMSE 1.897 → 1.869; R² 0.493 → 0.508). Both models show positive receiver split-half Pearson stability: 0.431 for static ridge and 0.452 for dynamic ridge, each with wide bootstrap intervals. The paired dynamic-minus-static Pearson difference is only +0.021, and its interval includes zero; Spearman ranking stability does not improve. These results do not show that dynamic features make receiver rankings more reliable.
#
# Route-level OOF scores answer whether dynamic tracking helps predict separation change on unseen games. Receiver split-half correlations answer whether a player's residual average carries across independent samples of games. This is a within-season check on the first eight weeks of 2021; it does not establish year-to-year persistence, and a positive correlation still needs replication on another season.
