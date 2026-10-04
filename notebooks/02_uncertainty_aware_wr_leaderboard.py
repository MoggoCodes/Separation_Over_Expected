# %% [markdown]
# # Uncertainty-Aware WR Separation Leaderboard
#
# The baseline notebook showed that the wide receiver model is the cleanest first surface for this project. This notebook focuses on the player-ranking question:
#
# > Which wide receivers create or preserve more separation than expected, and how confident are we that the effect is not just small-sample noise?
#
# The route-level score is:
#
# ```text
# SOE_route = delta_sep_actual - delta_sep_expected
# ```
#
# The player summary reports the mean route-level SOE and an approximate 95% confidence interval:
#
# ```text
# lower_95_soe = mean_soe - 1.96 * standard_error
# ```
#
# The conservative leaderboard sorts by `lower_95_soe`.

# %%
from pathlib import Path
import csv
import sys

PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from separation_over_expected.reports import read_csv_rows

SUMMARY_PATH = PROJECT_ROOT / "data" / "processed" / "position_baselines" / "receiver_baseline_summary_wr.csv"
METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "position_baselines" / "baseline_metrics_wr.csv"

PROJECT_ROOT

# %% [markdown]
# ## Model Check
#
# The WR-specific model is fit only on WR route observations. That makes the comparison group cleaner than the pooled model because outside receivers, slot receivers, and WR motion/alignment patterns are no longer mixed with TE and RB route jobs.

# %%
metrics = read_csv_rows(METRICS_PATH)
[row for row in metrics if row["split"] == "test"]

# %% [markdown]
# The ridge context model is the first useful expected-separation model for WRs. It explains about half of holdout variation in snap-to-release separation change, while the group-mean baseline barely improves over the global mean.

# %%
for row in metrics:
    if row["split"] == "test":
        print(f"{row['model']:>20}  R2={row['r2']}  RMSE={row['rmse']}  MAE={row['mae']}")

# %% [markdown]
# ## Conservative Leaderboard
#
# Sorting by `lower_95_soe` gives a conservative view: players need both a positive average and enough route volume to separate from zero. This is the table I would show first in a portfolio writeup.

# %%
wr_summary = read_csv_rows(SUMMARY_PATH)
wr_summary[:15]

# %% [markdown]
# ## Raw Mean Leaderboard
#
# The raw mean leaderboard answers a different question: who had the highest average SOE, regardless of uncertainty? Comparing this with the conservative leaderboard shows which players are sample-size-sensitive.

# %%
raw_mean_leaders = sorted(wr_summary, key=lambda row: float(row["mean_soe"]), reverse=True)
raw_mean_leaders[:15]

# %% [markdown]
# ## Players Most Affected By Uncertainty
#
# A simple way to see the effect of uncertainty is to compare each player's raw-mean rank with his conservative rank.

# %%
def add_rank(rows, key):
    ranked = sorted(rows, key=lambda row: float(row[key]), reverse=True)
    return {row["nflId"]: rank for rank, row in enumerate(ranked, start=1)}

mean_rank = add_rank(wr_summary, "mean_soe")
lower_rank = add_rank(wr_summary, "lower_95_soe")

rank_changes = []
for row in wr_summary:
    rank_changes.append({
        "displayName": row["displayName"],
        "routes": row["routes"],
        "mean_soe": row["mean_soe"],
        "lower_95_soe": row["lower_95_soe"],
        "mean_rank": mean_rank[row["nflId"]],
        "conservative_rank": lower_rank[row["nflId"]],
        "rank_drop": lower_rank[row["nflId"]] - mean_rank[row["nflId"]],
    })

sorted(rank_changes, key=lambda row: row["rank_drop"], reverse=True)[:15]

# %% [markdown]
# ## Takeaway
#
# The uncertainty-aware view changes the story in the right direction. Small-sample players can still appear if their signal is strong, but larger-sample receivers move up when their positive SOE is more stable. For the next iteration, the natural validation step is split-half stability: do WRs who rate well in weeks 1-4 continue to rate well in weeks 5-8?
