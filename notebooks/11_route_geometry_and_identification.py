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
# # Route Geometry and Identification
#
# This experiment measures each WR's observed, field-normalized path from snap through pass release. The continuous geometry features condition the separation benchmark on the realized route shape, so this is retrospective, not a snap-time prediction. The dataset marks who ran a route but contains no canonical route-type label.
#
# Separately, we cluster 11-point, start-relative path curves for visualization only. Cluster IDs are latent shape groups; they are neither named football routes nor predictors in the SOE model.

# %%
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from separation_over_expected.feature_schema import ROUTE_GEOMETRY_FEATURES, ROUTE_SHAPE_COLUMN
from separation_over_expected.reports import read_csv_rows

EXPERIMENT_DIR = PROJECT_ROOT / "data" / "processed" / "dynamic_features" / "route_geometry"
ROUTES = read_csv_rows(EXPERIMENT_DIR / "route_level_snap_to_release_geometry.csv")
METRICS = read_csv_rows(EXPERIMENT_DIR / "cross_validation" / "oof_metrics_wr.csv")
RELIABILITY = read_csv_rows(EXPERIMENT_DIR / "cross_validation" / "receiver_oof_reliability_wr.csv")
CLUSTER_DIR = EXPERIMENT_DIR / "clusters"
CLUSTER_METRICS = read_csv_rows(CLUSTER_DIR / "route_family_metrics_wr.csv")
CLUSTER_SUMMARIES = read_csv_rows(CLUSTER_DIR / "route_family_summaries_wr.csv")
len(ROUTES), len(METRICS), len(RELIABILITY), len(CLUSTER_METRICS)

# %% [markdown]
# ## Data and feature checks
#
# The route table has one row per route runner. The geometric features use only that receiver's trajectory between the snap and release events; shape coordinates are relative to the snap point and sampled at 11 equally spaced time fractions.

# %%
feature_coverage = {
    "route_rows": len(ROUTES),
    "unique_route_keys": len({(row["gameId"], row["playId"], row["nflId"]) for row in ROUTES}),
    "geometry_features": len(ROUTE_GEOMETRY_FEATURES),
    "missing_by_feature": {
        feature: sum(row.get(feature, "") == "" for row in ROUTES)
        for feature in ROUTE_GEOMETRY_FEATURES
    },
    "shape_point_counts": sorted({len(json.loads(row[ROUTE_SHAPE_COLUMN])) for row in ROUTES}),
}
feature_coverage

# %% [markdown]
# ## Route-level out-of-fold prediction
#
# Each model predicts WR separation change on games excluded from its training fold. Static and dynamic results should match the previous experiment exactly; the geometry model adds ten trajectory summaries to the existing dynamic ridge model.

# %%
OOF = [row for row in METRICS if row["fold"] == "OOF_ALL"]
FOLD_RESULTS = [row for row in METRICS if row["fold"] != "OOF_ALL"]
OOF, FOLD_RESULTS

# %% [markdown]
# ## Receiver score reliability
#
# Reliability is the correlation of each eligible receiver's mean out-of-fold SOE between two disjoint, week-balanced game halves. These 115 receivers each have at least 20 WR routes per half. The paired bootstrap difference uses the same receivers for the two models.

# %%
RELIABILITY

# %% [markdown]
# ## Unlabeled route-family exploration
#
# K-means runs on the standardized 20 coordinates (10 relative depth/lateral pairs after the fixed origin) for candidate k values 3–10. Silhouette is averaged over four seeded fits; mean pairwise adjusted Rand index (ARI) reports assignment stability across those fits. The selected k maximizes mean silhouette. No canonical route names are assigned.

# %%
CLUSTER_METRICS

# %%
class SVG:
    def __init__(self, markup): self.markup = markup
    def _repr_svg_(self): return self.markup

palette = ["#1769aa", "#d05b22", "#27834a", "#8c4b9f", "#c49a00", "#117c83", "#b23b53", "#555555"]
width, height = 780, 500
left, top, plot_w, plot_h = 70, 45, 650, 370
x_min, x_max, y_min, y_max = 0.0, 22.0, -14.0, 20.0
def sx(x): return left + (x - x_min) / (x_max - x_min) * plot_w
def sy(y): return top + plot_h - (y - y_min) / (y_max - y_min) * plot_h

lines = [f'<rect width="{width}" height="{height}" fill="white"/>']
lines += [f'<line x1="{left}" y1="{sy(0)}" x2="{left+plot_w}" y2="{sy(0)}" stroke="#aaa"/>',
          f'<line x1="{sx(0)}" y1="{top}" x2="{sx(0)}" y2="{top+plot_h}" stroke="#aaa"/>']
for i, summary in enumerate(CLUSTER_SUMMARIES):
    color = palette[i % len(palette)]
    mean_path = json.loads(summary["mean_path"])
    rep_path = json.loads(summary["representative_path"])
    mean_points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in mean_path)
    rep_points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in rep_path)
    lines.append(f'<polyline points="{mean_points}" fill="none" stroke="{color}" stroke-width="4"/>')
    lines.append(f'<polyline points="{rep_points}" fill="none" stroke="{color}" stroke-width="1.5" stroke-dasharray="5,4" opacity="0.65"/>')
    lines.append(f'<text x="{left+plot_w+12}" y="{top+22+i*28}" fill="{color}" font-size="13">C{summary["cluster_id"]}: {summary["route_count"]} routes</text>')
lines += [f'<text x="{left+plot_w/2}" y="{height-18}" text-anchor="middle" font-size="13">Forward displacement from route start (yards)</text>',
          f'<text x="18" y="{top+plot_h/2}" transform="rotate(-90 18 {top+plot_h/2})" text-anchor="middle" font-size="13">Lateral displacement (yards)</text>']
cluster_paths = SVG(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">{"".join(lines)}</svg>')
cluster_paths

# %%
CLUSTER_SUMMARIES

# %% [markdown]
# ## Interpretation
#
# Geometry materially improves retrospective route-level prediction: OOF RMSE falls from 1.869 to 1.749 and R² rises from 0.508 to 0.569, with lower RMSE in all five folds. However, receiver split-half Pearson reliability falls from 0.452 to 0.262 (paired difference -0.190; 95% bootstrap interval -0.315 to 0.037). The interval includes zero, so the point estimate is concerning but uncertain; Spearman reliability also declines slightly with an interval spanning zero.
#
# The route-family sweep selects k=4 by silhouette (about 0.409), but mean pairwise ARI is only about 0.60. Cluster sizes are highly imbalanced and the curves mainly separate broad movement patterns and route extent. Treat these as exploratory visual groupings, not dependable route labels.
#
# Because the geometry includes the full realized path through release, the prediction gain is not evidence of a better snap-time forecast. Conditioning on route execution can also remove between-route receiver signal from SOE. Keep geometry and clusters experimental; the existing dynamic model remains the preferred player-evaluation score pending further temporal and season-level validation.
