# %% [markdown]
# # Coverage Context Features
#
# The first WR model used nearest-defender geometry at the snap. That is a useful baseline, but it is too crude for zone coverage because the nearest defender is not always the defender who matters most.
#
# This notebook evaluates a richer defensive-context feature set:
#
# - nearest, second-nearest, and third-nearest defender distance at the snap
# - defenders within 3, 5, and 10 yards
# - nearest DB and nearest LB distance
# - nearest-defender depth and width leverage
# - defenders ahead of the receiver within 10 yards
# - close defenders inside/outside the receiver
#
# The question is whether this richer context improves WR prediction and stability.

# %%
from pathlib import Path
import csv
import statistics
import sys

PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from separation_over_expected.reports import pearson_correlation, read_csv_rows

ROUTE_TABLE = PROJECT_ROOT / "data" / "processed" / "route_level_snap_to_release.csv"
WR_METRICS = PROJECT_ROOT / "data" / "processed" / "position_baselines" / "baseline_metrics_wr.csv"
STABILITY = PROJECT_ROOT / "data" / "processed" / "stability" / "wr_split_half_stability.csv"

PROJECT_ROOT

# %% [markdown]
# ## New Columns
#
# The rebuilt route table has additional defender-context columns. These are now included in the ridge context model.

# %%
with ROUTE_TABLE.open(newline="") as f:
    reader = csv.DictReader(f)
    columns = reader.fieldnames

[column for column in columns if "defender" in column and column.endswith("_snap")]

# %% [markdown]
# ## WR Test Metrics
#
# The original WR ridge model had:
#
# ```text
# test R2   = 0.503
# test RMSE = 1.848
# test MAE  = 1.393
# ```
#
# The richer coverage-context model keeps essentially the same test R2 and RMSE, with a small MAE improvement.

# %%
metrics = read_csv_rows(WR_METRICS)
[row for row in metrics if row["split"] == "test"]

# %% [markdown]
# ## Split-Half Stability
#
# The original WR split-half stability correlation was 0.420. After adding these coverage-context features, the correlation is slightly lower.

# %%
stability = read_csv_rows(STABILITY)
early = [float(row["early_mean_soe"]) for row in stability]
late = [float(row["late_mean_soe"]) for row in stability]
correlation = pearson_correlation(early, late)

print(f"Qualified WRs: {len(stability)}")
print(f"Early/late Pearson correlation: {correlation:.3f}")

# %%
stability[:15]

# %% [markdown]
# ## Takeaway
#
# This feature set did not materially improve the WR model. Test error is almost unchanged, and split-half stability moved from 0.420 to about 0.406.
#
# That does not mean defensive context is unimportant. It means these simple density/leverage features are not enough to solve the defender-assignment problem. The next improvement should probably focus on one of two directions:
#
# 1. Add route-shape or route-phase information so defenders are interpreted relative to the receiver's actual path.
# 2. Build a better coverage-responsibility proxy, such as weighted defender distance over time rather than snap-only density.
