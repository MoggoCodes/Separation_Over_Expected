# %% [markdown]
# # WR Split-Half Stability
#
# A leaderboard is only useful if it measures something that persists. This notebook asks whether wide receivers who create more separation than expected early in the sample also tend to do so later.
#
# We use the WR-specific route-level predictions and compare:
#
# ```text
# weeks 1-4 mean SOE
# vs.
# weeks 5-8 mean SOE
# ```
#
# The goal is not a perfect correlation. Route running is noisy, opponent-dependent, and role-dependent. The question is whether the metric contains repeatable signal.

# %%
from pathlib import Path
import statistics
import sys

PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from separation_over_expected.reports import pearson_correlation, read_csv_rows, split_half_stability_rows

PREDICTIONS = PROJECT_ROOT / "data" / "processed" / "position_baselines" / "route_level_baseline_predictions_wr.csv"
STABILITY = PROJECT_ROOT / "data" / "processed" / "stability" / "wr_split_half_stability.csv"

PROJECT_ROOT

# %% [markdown]
# ## Build the Stability Table
#
# Each row is a WR with at least 20 routes in weeks 1-4 and at least 20 routes in weeks 5-8.

# %%
routes = read_csv_rows(PREDICTIONS)
stability = split_half_stability_rows(routes, position="WR", min_routes_per_half=20)
len(stability)

# %%
stability[:15]

# %% [markdown]
# ## Early vs Late Correlation
#
# The basic test is Pearson correlation between early mean SOE and late mean SOE.

# %%
early = [float(row["early_mean_soe"]) for row in stability]
late = [float(row["late_mean_soe"]) for row in stability]
correlation = pearson_correlation(early, late)
correlation

# %%
print(f"Qualified WRs: {len(stability)}")
print(f"Early mean SOE average: {statistics.fmean(early):.3f}")
print(f"Late mean SOE average: {statistics.fmean(late):.3f}")
print(f"Early/late Pearson correlation: {correlation:.3f}")

# %% [markdown]
# ## Late-Period Leaders
#
# The late-period leaderboard helps identify players whose later SOE supports the early signal and players who surged later.

# %%
late_leaders = sorted(stability, key=lambda row: float(row["late_mean_soe"]), reverse=True)
late_leaders[:15]

# %% [markdown]
# ## Biggest Changes
#
# Large changes are useful audit cases. Some are real role changes, some are sample noise, and some may expose model limitations.

# %%
improvers = sorted(stability, key=lambda row: float(row["late_minus_early"]), reverse=True)
decliners = sorted(stability, key=lambda row: float(row["late_minus_early"]))
improvers[:10]

# %%
decliners[:10]

# %% [markdown]
# ## Takeaway
#
# With the current baseline, WR SOE has a split-half correlation around 0.42 among qualifying receivers. That is a promising amount of repeatable signal for a first tracking-data metric, but it is not strong enough to treat the leaderboard as a final player ranking.
#
# The next modeling improvements should try to reduce route/context noise:
#
# 1. Add route-shape clusters so verticals, crossers, screens, and quick outs are not over-pooled.
# 2. Improve defender context beyond nearest defender.
# 3. Compare stability before and after those improvements.
