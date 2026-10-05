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
# # Pre-Release Quarterback Pocket Context
#
# Does observable quarterback movement and pass-rush proximity before release help predict receiver separation creation? The target remains `delta_sep = sep_release - sep_snap`. We compare the existing static ridge model, its defender-motion extension, and a third ridge model that adds pocket-context features. All models use the same five game-grouped folds stratified within week; every route is scored out of fold.
#
# The new inputs summarize QB movement and PFF-tagged pass-rusher distance from snap up to, but excluding, the release frame. They are contextual predictors, not an attribution of pressure or route quality.

# %%
from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from separation_over_expected.reports import read_csv_rows

EXPERIMENT_DIR = PROJECT_ROOT / "data" / "processed" / "dynamic_features" / "pocket_context"
ROUTES = read_csv_rows(EXPERIMENT_DIR / "route_level_snap_to_release_pocket.csv")
METRICS = read_csv_rows(EXPERIMENT_DIR / "cross_validation" / "oof_metrics_wr.csv")
RELIABILITY = read_csv_rows(EXPERIMENT_DIR / "cross_validation" / "receiver_oof_reliability_wr.csv")
len(ROUTES), len(METRICS), len(RELIABILITY)

# %% [markdown]
# ## Feature coverage and range checks
#
# The route table includes one row per route runner on eligible dropbacks. Check that pocket features are populated and inspect their broad distributions before interpreting model scores.

# %%
POCKET_FEATURES = [
    "qb_depth_drop_pre_release", "qb_lateral_drift_pre_release", "qb_path_length_pre_release",
    "qb_mean_speed_pre_release", "qb_max_speed_pre_release", "qb_mean_accel_pre_release",
    "qb_nearest_rusher_dist_snap", "qb_nearest_rusher_min_dist_pre_release",
    "qb_nearest_rusher_mean_dist_pre_release", "qb_rusher_closing_rate_pre_release",
    "qb_rusher_within_3yd_frame_share_pre_release", "qb_rusher_within_5yd_frame_share_pre_release",
    "qb_pressure_observed_fraction_pre_release", "pff_pass_rusher_count",
]
def feature_summary(rows, name):
    values = [float(row[name]) for row in rows if row.get(name, "") != ""]
    return {"feature": name, "missing": len(rows) - len(values), "min": round(min(values), 3),
            "mean": round(sum(values) / len(values), 3), "max": round(max(values), 3)}

coverage = [feature_summary(ROUTES, name) for name in POCKET_FEATURES]
coverage

# %% [markdown]
# ## Out-of-fold route prediction
#
# All three models score the same 20,415 WR routes. The paired model comparisons are the key: pocket features must improve on the existing dynamic model to add value beyond defender motion already in the pipeline.

# %%
overall = [row for row in METRICS if row["fold"] == "OOF_ALL"]
overall

# %%
fold_results = [row for row in METRICS if row["fold"] != "OOF_ALL"]
fold_results

# %% [markdown]
# ## Receiver-level reliability
#
# We split the out-of-fold predictions into two disjoint, week-balanced game sets and correlate each receiver's average SOE between halves. The paired bootstrap resamples the same 115 receivers for both models. Route prediction and receiver reliability answer different questions: a small average route error gain need not mean player evaluations become more repeatable.

# %%
RELIABILITY

# %% [markdown]
# ## Takeaway
#
# Pocket context gives a tiny additional route-level gain over dynamic defender context: OOF RMSE moves from 1.869 to 1.867 and R² from 0.508 to 0.509. The direction of the RMSE change is consistent across most folds, but its magnitude is negligible.
#
# Receiver reliability moves the other way. Split-half Pearson correlation is 0.424 for the pocket model versus 0.452 for dynamic context; the paired difference is -0.028 (95% bootstrap interval -0.050 to -0.011). Spearman ranking correlation also falls from 0.209 to 0.184, though the paired interval includes zero. On this evaluation, the new features do not improve receiver evaluation and may make level estimates less repeatable.
#
# Recommendation: keep the pocket features as an experiment, not as the preferred model inputs. The route-level improvement is too small to justify the reduction in Pearson receiver stability. Before trying a more complex model, inspect the added feature coefficients/regularization and test a narrower pressure feature set (for example, rusher proximity and closing only) against the exact same folds. This is a within-season, eight-week result; it does not establish year-to-year stability.
