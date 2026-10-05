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
# # Pressure-Only Context Features
#
# The broad pocket-context experiment added QB movement along with pressure indicators. This follow-up isolates six pressure features: nearest rusher distance at snap, minimum and mean distance before release, closing rate, and the share of pre-release frames within three and five yards. The target remains snap-to-release `delta_sep`.
#
# We compare static ridge, dynamic defender-context ridge, pressure-only ridge, and the broad pocket ridge. The same five game-grouped folds and week-balanced receiver reliability halves are used for each model.

# %%
from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from separation_over_expected.feature_schema import PRESSURE_CONTEXT_FEATURES
from separation_over_expected.reports import read_csv_rows

EXPERIMENT_DIR = PROJECT_ROOT / "data" / "processed" / "dynamic_features" / "pressure_context" / "cross_validation"
METRICS = read_csv_rows(EXPERIMENT_DIR / "oof_metrics_wr.csv")
RELIABILITY = read_csv_rows(EXPERIMENT_DIR / "receiver_oof_reliability_wr.csv")
FEATURES = PRESSURE_CONTEXT_FEATURES
FEATURES

# %% [markdown]
# ## Out-of-fold route prediction
#
# The OOF summary aggregates the same 20,415 routes scored by models trained on other games. Compare pressure-only with dynamic context to see whether pressure adds route-level signal, then compare broad pocket context with pressure-only to see whether QB movement adds anything beyond those pressure features.

# %%
OOF = [row for row in METRICS if row["fold"] == "OOF_ALL"]
OOF

# %%
FOLD_RESULTS = [row for row in METRICS if row["fold"] != "OOF_ALL"]
FOLD_RESULTS

# %% [markdown]
# ## Receiver-level reliability
#
# Each receiver needs at least 20 routes in both game halves. We report Pearson agreement in residual levels and Spearman agreement in rankings. The paired bootstrap compares the models on the same receivers.

# %%
RELIABILITY

# %% [markdown]
# ## Interpretation
#
# Pressure-only produces a small route-level gain over dynamic context (OOF RMSE 1.869 to 1.867; R² 0.508 to 0.509). Its receiver Pearson reliability is 0.434 versus 0.452 for dynamic context; the paired difference is -0.018 (95% interval -0.035 to 0.005), so this evaluation neither shows a reliability gain nor clear evidence of a loss. Spearman reliability is also slightly lower, with a paired interval spanning zero.
#
# The broad pocket model has effectively the same route RMSE as pressure-only, but lower receiver reliability (Pearson 0.424; Spearman 0.184). Relative to pressure-only, the broad feature set's paired reliability differences are -0.010 Pearson (95% interval -0.024 to -0.003) and -0.020 Spearman (95% interval -0.040 to -0.002). The QB movement additions therefore do not help this receiver-stability check.
#
# Takeaway: pressure-only is a more promising and narrower predictor than the broad pocket set, but it does not beat the existing dynamic model on receiver reliability. Keep dynamic context as the current preferred model. The modest route-prediction gain alone is not enough to claim better player evaluation. Results cover only the first eight weeks of one season.
