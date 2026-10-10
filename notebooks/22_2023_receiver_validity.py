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
# # Does Cross-Season Receiver Repeatability Survive Basic Stress Tests?
#
# This is a bounded validity check, not a model-tuning exercise. We hold the endpoint-nearest separation target and the frozen shared-feature Ridge predictions fixed. We vary only the minimum route/game eligibility rule, then check how much the receiver-level Pearson correlation depends on individual players. The data compare game-held-out 2021 routes with 2023 routes scored by a model fit on 2021.
#
# Interpretation is deliberately narrow: a repeatable residual may reflect receiver skill, role, team, route assignment, or measurement differences. It is not an isolated causal estimate of receiver talent. The 2021 sample covers Weeks 1–8; 2023 uses a snap-to-release-like window, so season and data-window differences remain.

# %%
from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from IPython.display import display
from separation_over_expected.receiver_validity import audit_cross_season_receiver_validity
from separation_over_expected.reports import read_csv_rows, write_csv

PREDICTIONS = PROJECT_ROOT / "data/processed/cross_season_receiver_reliability/route_predictions.csv"
OUTPUT = PROJECT_ROOT / "data/processed/cross_season_receiver_validity"
if not PREDICTIONS.exists():
    raise FileNotFoundError(
        "Run `uv run separation-over-expected cross-season-receiver-reliability` first."
    )
predictions = read_csv_rows(PREDICTIONS)
len(predictions)

# %% [markdown]
# ## Pre-set eligibility sensitivity
#
# The primary cohort remains 20 routes in five games per season. The adjacent cohorts test a lower route threshold, a higher route threshold, and stricter game coverage. The same receivers and predictions are not forced across cohorts; cohort size is reported so changes in correlation can be interpreted alongside selection.

# %%
cohorts, receiver_summaries, leave_one_out = audit_cross_season_receiver_validity(
    predictions,
    bootstrap_samples=500,
    seed=42,
)
OUTPUT.mkdir(parents=True, exist_ok=True)
write_csv(OUTPUT / "cohort_sensitivity.csv", list(cohorts[0]), cohorts)
write_csv(OUTPUT / "receiver_summaries.csv", list(receiver_summaries[0]), receiver_summaries)
write_csv(OUTPUT / "leave_one_receiver_out.csv", list(leave_one_out[0]), leave_one_out)

cohort_view = [
    {
        "routes / games": f'{row["min_routes"]} / {row["min_games"]}',
        "eligible receivers": row["eligible_receivers"],
        "model": row["comparison"],
        "Pearson r": round(float(row["pearson"]), 3),
        "95% interval": f'[{float(row["pearson_ci_lower"]):.3f}, {float(row["pearson_ci_upper"]):.3f}]',
        "Spearman ρ": round(float(row["spearman"]), 3),
    }
    for row in cohorts
]
display(cohort_view)

# %% [markdown]
# ## Influence of individual receivers
#
# Leave-one-out correlation is a stress test, not a way to discard inconvenient players. Large swings mean the cohort-level estimate is too dependent on a few observations to support broad claims. The named pair was selected from the earlier audit because removing both had already been observed to sharply reduce dynamic Pearson correlation; all individual omissions are also included.

# %%
influence_view = [
    {
        "model": row["model"],
        "omitted receiver": row["omitted_receiver"],
        "receivers remaining": int(row["receivers_remaining"]),
        "Pearson r": round(float(row["pearson"]), 3),
        "change vs full": round(float(row["change_vs_full"]), 3),
    }
    for row in leave_one_out
    if row["omitted_receiver"] == "FULL COHORT"
    or row["omitted_receiver"] == "Isaiah McKenzie + Rondale Moore"
    or abs(float(row["change_vs_full"])) >= 0.10
]
display(influence_view)

# %% [markdown]
# ## Decision
#
# The project should continue as a careful route-level measurement and validation study, but these results determine whether we can promote the player-level metric as reliable. If the association changes materially across cohorts or drops sharply when a small number of players are omitted, the short-term conclusion is that receiver rankings are not yet robust. The next productive step would then be to improve measurement comparability or uncertainty estimation—not to tune features against this small cross-season correlation.

# %%
dynamic_rows = [row for row in cohorts if row["comparison"] == "dynamic"]
dynamic_influence = [row for row in leave_one_out if row["model"] == "dynamic"]
decision = {
    "cohort_dynamic_pearson_range": (
        min(float(row["pearson"]) for row in dynamic_rows),
        max(float(row["pearson"]) for row in dynamic_rows),
    ),
    "lowest_dynamic_leave_one_out": min(float(row["pearson"]) for row in dynamic_influence),
    "highest_dynamic_leave_one_out": max(float(row["pearson"]) for row in dynamic_influence),
    "interpretation": "Treat receiver-level repeatability as tentative unless it is stable to cohort and player influence checks.",
}
decision
