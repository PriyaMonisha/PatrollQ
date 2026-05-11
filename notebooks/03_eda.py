# ================================================================
# PATROLIQ — EDA v1 (Chicago Crime Analytics)
# notebooks/02_eda.py
#
# CHARTS PRODUCED (saved to docs/figures/):
#   Chart 1 — Crime Type Distribution      → 02_crime_type_distribution.png
#   Chart 2 — Geographic Crime Patterns    → 02_geographic_scatter.png
#   Chart 3 — Hourly Crime Heatmap         → 02_hourly_heatmap.png
#   Chart 4 — Monthly & Seasonal Trends    → 02_monthly_trend.png
#   Chart 5 — Arrest Rate Analysis         → 02_arrest_rates.png
#   Chart 6 — Domestic Incident Analysis   → 02_domestic_breakdown.png
#   Chart 7 — Crime Severity Distribution  → 02_severity_distribution.png
#
# KDE METHOD: scipy gaussian_kde (direct x/y arrays)
#   Same approach as EMI EDA v3 — no seaborn internal line bugs
# ================================================================

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import gaussian_kde

from config import (
    DATA_PROCESSED_DIR,
    DOCS_FIGURES_DIR,
    FAST_MODE,
    PROCESSED_CSV,
    RANDOM_STATE,
    SEVERITY_SCORES,
    SEVERITY_DEFAULT,
)

# ── Global style ─────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi'        : 130,
    'font.size'         : 10,
    'axes.titlesize'    : 12,
    'axes.titleweight'  : 'bold',
    'axes.labelsize'    : 10,
    'xtick.labelsize'   : 9,
    'ytick.labelsize'   : 9,
    'legend.fontsize'   : 9,
    'figure.titlesize'  : 14,
    'figure.titleweight': 'bold',
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
})

# ================================================================
# MASTER PALETTE
# ================================================================
D2   = sns.color_palette('Dark2_r',  8)
PR   = sns.color_palette('Paired_r', 12)
AC   = sns.color_palette('Accent_r', 8)
T20  = sns.color_palette('tab20',    20)  # for 33 crime types

# Binary: Arrested = teal-green, Not Arrested = orange-red
ARR_CLR  = D2[7]   # teal-green (#1b9e77)
NARR_CLR = D2[6]   # orange-red (#d95f02)
BAR_2    = [NARR_CLR, ARR_CLR]
DOM_CLR  = D2[0]   # domestic
NDOM_CLR = D2[3]   # non-domestic
DARK     = '#2D3436'

# Day and month labels (pandas: 0=Monday, 6=Sunday)
DAY_NAMES   = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# ── Paths ─────────────────────────────────────────────────────
DOCS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(name: str) -> None:
    """Save current figure to docs/figures/ and close."""
    path = DOCS_FIGURES_DIR / name
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {name}")


# ================================================================
# HELPER: scipy KDE (same robust approach as EMI EDA v3)
# ================================================================
def kde_fill(ax, data, color, label, clip_lo=0,
             clip_hi=None, linestyle='-', alpha_fill=0.20,
             lw=2.5, n_points=500):
    """
    Compute KDE with scipy directly — no seaborn internal line bugs.
    Draws filled area + line on ax.
    clip_lo/clip_hi hard-clamp the x range.
    """
    data = pd.Series(data).dropna()
    if len(data) < 10:
        return

    hi   = float(clip_hi) if clip_hi is not None else float(data.quantile(0.995))
    lo   = float(clip_lo)
    xgrd = np.linspace(lo, hi, n_points)

    try:
        kde  = gaussian_kde(data, bw_method='scott')
        ygrd = kde(xgrd)
        ygrd[xgrd < lo] = 0.0
    except Exception:
        return

    ax.plot(xgrd, ygrd, color=color, linewidth=lw,
            linestyle=linestyle, label=label, zorder=3)
    ax.fill_between(xgrd, ygrd, alpha=alpha_fill, color=color, zorder=2)


# ================================================================
# LOAD DATA
# ================================================================
print("=" * 65)
print("   PATROLIQ — EDA v1 (Chicago Crime Analytics)")
print("=" * 65)

# FAST_MODE: load dev file if it exists, else fall back to production file
if FAST_MODE:
    dev_csv = DATA_PROCESSED_DIR / "chicago_crime_dev.csv.gz"
    csv_path = dev_csv if dev_csv.exists() else PROCESSED_CSV
else:
    csv_path = PROCESSED_CSV

if not csv_path.exists():
    print(f"\nERROR: Processed data not found: {csv_path}")
    print("  Run notebooks/01_data_acquisition.py first.")
    sys.exit(1)

print(f"\nLoading: {csv_path.name}")
df = pd.read_csv(csv_path)

# Ensure bool dtypes for Arrest/Domestic (sometimes restored as object by CSV round-trip)
for col in ['arrest', 'domestic']:
    if col in df.columns and df[col].dtype != bool:
        df[col] = df[col].astype(str).str.lower().map(
            {'true': True, 'false': False, '1': True, '0': False}
        ).fillna(False).astype(bool)

# Ensure int dtypes for temporal columns
for col in ['Hour', 'Day_of_Week', 'Month', 'Year']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

total = len(df)
print(f"Loaded: {total:,} rows x {df.shape[1]} columns")
print(f"  Memory : {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# ── Compute Crime_Severity_Score inline for EDA ───────────────
# (Official computation happens in engineer.py; this is for EDA only)
if 'primary_type' in df.columns:
    df['Crime_Severity_Score'] = (
        df['primary_type'].map(SEVERITY_SCORES).fillna(SEVERITY_DEFAULT).astype(int)
    )

# Scatter plot sample — 20K for visual clarity
SCATTER_N = min(20_000, total)
df_scatter = df.sample(SCATTER_N, random_state=RANDOM_STATE)

print(f"  Crime types : {df['primary_type'].nunique()}")
print(f"  Year range  : {int(df['Year'].min())} - {int(df['Year'].max())}")
print(f"  Arrest rate : {df['arrest'].mean()*100:.1f}%")
print(f"  Domestic %  : {df['domestic'].mean()*100:.1f}%")
print("=" * 65)


# ================================================================
# CHART 1: CRIME TYPE DISTRIBUTION
# ================================================================
print("\n[Chart 1] Crime Type Distribution...")

crime_counts = df['primary_type'].value_counts()
top15        = crime_counts.head(15)
top10        = crime_counts.head(10)
others_count = crime_counts.iloc[10:].sum()

fig, axes = plt.subplots(1, 3, figsize=(22, 8))
fig.suptitle('Chicago Crime Type Distribution', y=0.98)

# ── 1a: Top 15 horizontal bar ─────────────────────────────────
colors_15 = [T20[i % 20] for i in range(len(top15))]
bars = axes[0].barh(top15.index[::-1], top15.values[::-1],
                    color=colors_15[::-1], edgecolor='white',
                    alpha=0.92, height=0.75)
axes[0].set_title('Top 15 Crime Types')
axes[0].set_xlabel('Number of Crimes')
for bar, val in zip(bars, top15.values[::-1]):
    axes[0].text(
        bar.get_width() + top15.max() * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f'{val:,}', va='center', fontsize=8
    )
axes[0].set_xlim(0, top15.max() * 1.18)

# ── 1b: Top 10 + Others donut ─────────────────────────────────
donut_labels  = list(top10.index) + ['Others']
donut_vals    = list(top10.values) + [others_count]
donut_colors  = [T20[i % 20] for i in range(10)] + ['#b0b0b0']

wedge_kw = dict(width=0.44, edgecolor='white', linewidth=2)
axes[1].pie(
    donut_vals, labels=None, colors=donut_colors,
    startangle=90, wedgeprops=wedge_kw,
    autopct='%1.1f%%', pctdistance=0.78,
)
axes[1].text(0, 0.08, f'{total:,}', ha='center', va='center',
             fontsize=13, fontweight='bold', color=DARK)
axes[1].text(0, -0.14, 'Crimes', ha='center', va='center',
             fontsize=9, color='gray')
legend_patches = [
    mpatches.Patch(color=donut_colors[i], label=lbl[:22])
    for i, lbl in enumerate(donut_labels)
]
axes[1].legend(handles=legend_patches, loc='lower center',
               bbox_to_anchor=(0.5, -0.28), ncol=2,
               fontsize=7.5, framealpha=0)
axes[1].set_title('Share by Crime Type\n(Top 10 + Others)')

# ── 1c: Crimes per year (bar) ─────────────────────────────────
year_counts = df['Year'].value_counts().sort_index()
axes[2].bar(year_counts.index, year_counts.values,
            color=AC[2], edgecolor='white', alpha=0.90, width=0.75)
axes[2].set_title('Crimes by Year')
axes[2].set_xlabel('Year')
axes[2].set_ylabel('Number of Crimes')
axes[2].tick_params(axis='x', rotation=45)
axes[2].axhline(year_counts.mean(), color=DARK, linestyle='--',
                alpha=0.6, linewidth=1.5,
                label=f'Mean {year_counts.mean():,.0f}')
axes[2].legend()

plt.tight_layout(pad=2.0)
save_fig('02_crime_type_distribution.png')


# ================================================================
# CHART 2: GEOGRAPHIC CRIME PATTERNS
# ================================================================
print("\n[Chart 2] Geographic Crime Patterns...")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Geographic Crime Distribution — Chicago', y=0.98)

# ── 2a: Lat/lon scatter (severity-coded color) ────────────────
sc = axes[0, 0].scatter(
    df_scatter['longitude'], df_scatter['latitude'],
    c=df_scatter['Crime_Severity_Score'],
    cmap='YlOrRd', s=0.8, alpha=0.45, linewidths=0
)
cbar = plt.colorbar(sc, ax=axes[0, 0], shrink=0.8, pad=0.02)
cbar.set_label('Severity Score', fontsize=9)
axes[0, 0].set_title(f'Crime Locations (sample {SCATTER_N:,})\nColored by Severity Score')
axes[0, 0].set_xlabel('Longitude')
axes[0, 0].set_ylabel('Latitude')

# ── 2b: Crime count by district ───────────────────────────────
if 'district' in df.columns:
    dist_counts = (
        df.groupby('district').size().sort_values(ascending=False).head(20)
    )
    dist_colors = [AC[i % len(AC)] for i in range(len(dist_counts))]
    axes[0, 1].bar(
        dist_counts.index.astype(str), dist_counts.values,
        color=dist_colors, edgecolor='white', alpha=0.90
    )
    axes[0, 1].set_title('Crime Count by District (Top 20)')
    axes[0, 1].set_xlabel('District Number')
    axes[0, 1].set_ylabel('Number of Crimes')
    axes[0, 1].tick_params(axis='x', rotation=45)

# ── 2c: Arrest rate by district ───────────────────────────────
if 'district' in df.columns:
    dist_arrest = (
        df.groupby('district')['arrest']
        .agg(['mean', 'count'])
        .query('count >= 100')  # only districts with enough data
    )
    dist_arrest['pct'] = dist_arrest['mean'] * 100
    dist_arrest = dist_arrest.sort_values('pct', ascending=True).tail(20)
    overall_arr_pct = df['arrest'].mean() * 100

    cols_arr = [ARR_CLR if v >= overall_arr_pct else NARR_CLR
                for v in dist_arrest['pct']]
    bars = axes[1, 0].barh(
        dist_arrest.index.astype(str), dist_arrest['pct'],
        color=cols_arr, edgecolor='white', alpha=0.92
    )
    axes[1, 0].axvline(overall_arr_pct, color=DARK, linestyle='--',
                       linewidth=1.5, alpha=0.7,
                       label=f'Overall {overall_arr_pct:.1f}%')
    axes[1, 0].set_title('Arrest Rate by District (%)\n(green = above average)')
    axes[1, 0].set_xlabel('Arrest Rate (%)')
    axes[1, 0].set_ylabel('District')
    axes[1, 0].legend(fontsize=8)
    for bar, val in zip(bars, dist_arrest['pct']):
        axes[1, 0].text(
            bar.get_width() + 0.2,
            bar.get_y() + bar.get_height() / 2,
            f'{val:.1f}%', va='center', fontsize=8
        )

# ── 2d: Top 15 location types ─────────────────────────────────
if 'location_description' in df.columns:
    loc_counts = df['location_description'].value_counts().head(15)
    loc_colors = [PR[i % len(PR)] for i in range(len(loc_counts))]
    axes[1, 1].barh(loc_counts.index[::-1], loc_counts.values[::-1],
                    color=loc_colors[::-1], edgecolor='white',
                    alpha=0.90, height=0.75)
    axes[1, 1].set_title('Top 15 Crime Location Types')
    axes[1, 1].set_xlabel('Number of Crimes')
    axes[1, 1].tick_params(axis='y', labelsize=8)

plt.tight_layout(pad=2.2)
save_fig('02_geographic_scatter.png')


# ================================================================
# CHART 3: HOURLY CRIME HEATMAP (centerpiece visualization)
# ================================================================
print("\n[Chart 3] Hourly Crime Heatmap...")

# Pivot: rows = Day_of_Week (0-6), columns = Hour (0-23)
pivot = (
    df.groupby(['Day_of_Week', 'Hour'])
    .size()
    .unstack(fill_value=0)
)
# Ensure all 24 hours present (fill missing with 0)
for h in range(24):
    if h not in pivot.columns:
        pivot[h] = 0
pivot = pivot[sorted(pivot.columns)]
pivot.index = [DAY_NAMES[int(i)] for i in pivot.index]

fig = plt.figure(figsize=(22, 12))
fig.suptitle('When Crimes Happen — Hour × Day Heatmap', y=0.98)
gs_ = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.30)

ax_heat = fig.add_subplot(gs_[:, :2])  # big heatmap — left 2/3
ax_hour = fig.add_subplot(gs_[0, 2])   # top right: hourly bar
ax_day  = fig.add_subplot(gs_[1, 2])   # bottom right: day-of-week bar

# ── 3a: Main heatmap ──────────────────────────────────────────
sns.heatmap(
    pivot, cmap='YlOrRd',
    annot=True, fmt=',d', annot_kws={'size': 7.5},
    linewidths=0.3, linecolor='#f0f0f0',
    ax=ax_heat,
    cbar_kws={'shrink': 0.65, 'label': 'Crime Count', 'pad': 0.01}
)
ax_heat.set_title(
    'Crime Frequency by Hour × Day of Week\n'
    'Yellow = fewer crimes  |  Red = more crimes',
    fontsize=12, pad=12
)
ax_heat.set_xlabel('Hour of Day (0 = midnight, 12 = noon)', fontsize=10)
ax_heat.set_ylabel('Day of Week', fontsize=10)
ax_heat.tick_params(axis='x', rotation=0, labelsize=8.5)
ax_heat.tick_params(axis='y', rotation=0, labelsize=9)

# ── 3b: Crimes by hour ────────────────────────────────────────
hourly_counts = df.groupby('Hour').size()
peak_hour = int(hourly_counts.idxmax())
colors_hour = [
    ARR_CLR if h == peak_hour else AC[1]
    for h in hourly_counts.index
]
ax_hour.bar(hourly_counts.index, hourly_counts.values,
            color=colors_hour, edgecolor='white', alpha=0.90)
ax_hour.set_title(f'Crimes by Hour\n(Peak: {peak_hour:02d}:00)')
ax_hour.set_xlabel('Hour')
ax_hour.set_ylabel('Crime Count')
ax_hour.axvline(peak_hour, color=NARR_CLR, linestyle='--',
                linewidth=1.8, alpha=0.8)

# ── 3c: Crimes by day of week ─────────────────────────────────
day_counts = df.groupby('Day_of_Week').size()
day_colors = [
    AC[4] if i >= 5 else AC[1]
    for i in day_counts.index
]
ax_day.bar(
    [DAY_NAMES[int(i)] for i in day_counts.index],
    day_counts.values,
    color=day_colors, edgecolor='white', alpha=0.90
)
ax_day.set_title('Crimes by Day\n(Weekend highlighted)')
ax_day.set_xlabel('Day of Week')
ax_day.set_ylabel('Crime Count')
ax_day.legend(handles=[
    mpatches.Patch(color=AC[4], label='Weekend'),
    mpatches.Patch(color=AC[1], label='Weekday'),
], fontsize=8)

save_fig('02_hourly_heatmap.png')


# ================================================================
# CHART 4: MONTHLY & SEASONAL TRENDS
# ================================================================
print("\n[Chart 4] Monthly & Seasonal Trends...")

fig, axes = plt.subplots(2, 2, figsize=(18, 13))
fig.suptitle('Crime Temporal Patterns — Monthly & Seasonal', y=0.98)

# ── 4a: Monthly crime count (line per year, or single averaged line) ─
monthly_by_year = df.groupby(['Year', 'Month']).size().reset_index(name='count')
# Average monthly profile across all years
monthly_avg = monthly_by_year.groupby('Month')['count'].mean()

actual_months = monthly_avg.index  # only months present in data (FAST_MODE may have < 12)
month_labels  = [MONTH_NAMES[m - 1] for m in actual_months]
axes[0, 0].bar(actual_months, monthly_avg.values,
               color=AC[3], edgecolor='white', alpha=0.88, width=0.75)
axes[0, 0].plot(actual_months, monthly_avg.values,
                color=DARK, linewidth=2.0, marker='o',
                markersize=5, zorder=3, label='Avg crime count')
axes[0, 0].set_title('Average Monthly Crime Count\n(across all sampled years)')
axes[0, 0].set_xlabel('Month')
axes[0, 0].set_ylabel('Avg Crime Count')
axes[0, 0].set_xticks(actual_months)
axes[0, 0].set_xticklabels(month_labels, rotation=45)
axes[0, 0].legend()
peak_month = int(monthly_avg.idxmax())
axes[0, 0].annotate(
    f'Peak:\n{MONTH_NAMES[peak_month-1]}',
    xy=(peak_month, monthly_avg.max()),
    xytext=(peak_month + 1.2, monthly_avg.max() * 0.95),
    fontsize=8, color=NARR_CLR,
    arrowprops=dict(arrowstyle='->', color=NARR_CLR, lw=1.4)
)

# ── 4b: Year-over-year total crime count ──────────────────────
year_counts = df.groupby('Year').size()
axes[0, 1].bar(year_counts.index, year_counts.values,
               color=AC[1], edgecolor='white', alpha=0.90, width=0.75)
axes[0, 1].plot(year_counts.index, year_counts.values,
                color=DARK, linewidth=2.0, marker='o', markersize=5, zorder=3)
axes[0, 1].set_title('Crime Count by Year')
axes[0, 1].set_xlabel('Year')
axes[0, 1].set_ylabel('Number of Crimes')
axes[0, 1].tick_params(axis='x', rotation=45)

# ── 4c: Seasonal breakdown ────────────────────────────────────
if 'Season' in df.columns:
    season_order = ['Spring', 'Summer', 'Fall', 'Winter']
    season_counts = df['Season'].value_counts().reindex(season_order, fill_value=0)
    season_colors = [AC[i] for i in range(len(season_counts))]
    bars = axes[1, 0].bar(season_counts.index, season_counts.values,
                          color=season_colors, edgecolor='white', alpha=0.92)
    axes[1, 0].set_title('Crimes by Season')
    axes[1, 0].set_xlabel('Season')
    axes[1, 0].set_ylabel('Number of Crimes')
    for bar, val in zip(bars, season_counts.values):
        pct = val / total * 100
        axes[1, 0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.003,
            f'{val:,}\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=9, fontweight='bold'
        )

# ── 4d: Weekend vs weekday comparison ─────────────────────────
if 'Is_Weekend' in df.columns:
    wd_counts = df.groupby('Is_Weekend').size()
    labels = ['Weekday', 'Weekend']
    vals   = [wd_counts.get(False, 0), wd_counts.get(True, 0)]
    pcts   = [v / total * 100 for v in vals]

    bars = axes[1, 1].bar(labels, vals,
                          color=[AC[1], AC[4]], edgecolor='white',
                          alpha=0.92, width=0.50)
    axes[1, 1].set_title('Weekday vs Weekend Crime Count')
    axes[1, 1].set_ylabel('Number of Crimes')
    for bar, val, pct in zip(bars, vals, pcts):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.003,
            f'{val:,}\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=10, fontweight='bold'
        )
    axes[1, 1].set_ylim(0, max(vals) * 1.20)

plt.tight_layout(pad=2.5)
save_fig('02_monthly_trend.png')


# ================================================================
# CHART 5: ARREST RATE ANALYSIS
# ================================================================
print("\n[Chart 5] Arrest Rate Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(18, 13))
fig.suptitle('Arrest Rate Analysis', y=0.98)

overall_arr_pct = df['arrest'].mean() * 100

# ── 5a: Arrest rate by crime type (top 15 sorted) ─────────────
arr_by_type = (
    df.groupby('primary_type')['arrest']
    .agg(['mean', 'count'])
    .query('count >= 50')
)
arr_by_type['pct'] = arr_by_type['mean'] * 100
arr_by_type = arr_by_type.sort_values('pct', ascending=True).tail(20)

col_arr_type = [
    ARR_CLR if v >= overall_arr_pct else NARR_CLR
    for v in arr_by_type['pct']
]
bars = axes[0, 0].barh(arr_by_type.index, arr_by_type['pct'],
                       color=col_arr_type, edgecolor='white', alpha=0.92)
axes[0, 0].axvline(overall_arr_pct, color=DARK, linestyle='--',
                   linewidth=1.5, alpha=0.7,
                   label=f'Overall {overall_arr_pct:.1f}%')
axes[0, 0].set_title('Arrest Rate by Crime Type (%)\n(green = above average, top 20)')
axes[0, 0].set_xlabel('Arrest Rate (%)')
axes[0, 0].legend(fontsize=8)
for bar, val in zip(bars, arr_by_type['pct']):
    axes[0, 0].text(
        bar.get_width() + 0.3,
        bar.get_y() + bar.get_height() / 2,
        f'{val:.1f}%', va='center', fontsize=8
    )
axes[0, 0].tick_params(axis='y', labelsize=7.5)

# ── 5b: Arrest rate by hour of day ────────────────────────────
arr_by_hour = df.groupby('Hour')['arrest'].mean() * 100
axes[0, 1].plot(arr_by_hour.index, arr_by_hour.values,
                color=ARR_CLR, linewidth=2.5, marker='o',
                markersize=5, zorder=3)
axes[0, 1].fill_between(arr_by_hour.index, arr_by_hour.values,
                        alpha=0.20, color=ARR_CLR)
axes[0, 1].axhline(overall_arr_pct, color=DARK, linestyle='--',
                   linewidth=1.5, alpha=0.6,
                   label=f'Overall {overall_arr_pct:.1f}%')
axes[0, 1].set_title('Arrest Rate by Hour of Day')
axes[0, 1].set_xlabel('Hour (0=midnight, 12=noon)')
axes[0, 1].set_ylabel('Arrest Rate (%)')
axes[0, 1].legend()
axes[0, 1].set_xlim(0, 23)

# ── 5c: Arrest count (arrested vs not arrested) ───────────────
arr_counts = df['arrest'].value_counts()
n_arrested = int(arr_counts.get(True, 0))
n_not      = int(arr_counts.get(False, 0))

bars = axes[1, 0].bar(
    ['Not Arrested', 'Arrested'], [n_not, n_arrested],
    color=[NARR_CLR, ARR_CLR], edgecolor='white',
    alpha=0.93, width=0.45
)
axes[1, 0].set_title('Arrest Count Overview')
axes[1, 0].set_ylabel('Number of Crimes')
axes[1, 0].set_ylim(0, max(n_not, n_arrested) * 1.22)
for bar, val in zip(bars, [n_not, n_arrested]):
    pct = val / total * 100
    axes[1, 0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(n_not, n_arrested) * 0.015,
        f'{val:,}\n({pct:.1f}%)',
        ha='center', va='bottom', fontsize=11, fontweight='bold', color=DARK
    )

# ── 5d: Arrest rate by season ─────────────────────────────────
if 'Season' in df.columns:
    season_order = ['Spring', 'Summer', 'Fall', 'Winter']
    arr_season = (
        df.groupby('Season')['arrest'].mean() * 100
    ).reindex(season_order, fill_value=0)

    seas_colors = [
        ARR_CLR if v >= overall_arr_pct else NARR_CLR
        for v in arr_season.values
    ]
    bars = axes[1, 1].bar(arr_season.index, arr_season.values,
                          color=seas_colors, edgecolor='white', alpha=0.92)
    axes[1, 1].axhline(overall_arr_pct, color=DARK, linestyle='--',
                       linewidth=1.5, alpha=0.6,
                       label=f'Overall {overall_arr_pct:.1f}%')
    axes[1, 1].set_title('Arrest Rate by Season')
    axes[1, 1].set_ylabel('Arrest Rate (%)')
    axes[1, 1].legend(fontsize=8)
    for bar, val in zip(bars, arr_season.values):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f'{val:.1f}%', ha='center', va='bottom',
            fontsize=10, fontweight='bold'
        )

plt.tight_layout(pad=2.5)
save_fig('02_arrest_rates.png')


# ================================================================
# CHART 6: DOMESTIC INCIDENT ANALYSIS
# ================================================================
print("\n[Chart 6] Domestic Incident Analysis...")

dom_df  = df[df['domestic'] == True]
ndom_df = df[df['domestic'] == False]
n_dom   = len(dom_df)
n_ndom  = len(ndom_df)
dom_pct = n_dom / total * 100

fig, axes = plt.subplots(2, 2, figsize=(18, 13))
fig.suptitle('Domestic Incident Analysis', y=0.98)

# ── 6a: Domestic vs non-domestic donut ───────────────────────
wedge_kw = dict(width=0.44, edgecolor='white', linewidth=3)
patches, texts, autotexts = axes[0, 0].pie(
    [n_ndom, n_dom],
    colors=[NDOM_CLR, DOM_CLR],
    startangle=90,
    wedgeprops=wedge_kw,
    autopct='%1.1f%%', pctdistance=0.78,
    labels=['Non-Domestic', 'Domestic'],
    labeldistance=1.12
)
for t in texts:
    t.set_fontsize(9)
    t.set_fontweight('bold')
for at in autotexts:
    at.set_fontsize(9)
    at.set_color('white')
    at.set_fontweight('bold')
axes[0, 0].text(0, 0.08, f'{total:,}', ha='center', va='center',
                fontsize=13, fontweight='bold', color=DARK)
axes[0, 0].text(0, -0.14, 'Crimes', ha='center', va='center',
                fontsize=9, color='gray')
axes[0, 0].legend(handles=[
    mpatches.Patch(color=NDOM_CLR, label=f'Non-Domestic: {n_ndom:,}'),
    mpatches.Patch(color=DOM_CLR,  label=f'Domestic: {n_dom:,}'),
], loc='lower center', bbox_to_anchor=(0.5, -0.08),
   fontsize=8.5, framealpha=0, ncol=1)
axes[0, 0].set_title('Domestic vs Non-Domestic Crimes')

# ── 6b: Top 10 crime types with high domestic rate ───────────
dom_rate_by_type = (
    df.groupby('primary_type')['domestic']
    .agg(['mean', 'count'])
    .query('count >= 50')
)
dom_rate_by_type['pct'] = dom_rate_by_type['mean'] * 100
dom_rate_by_type = (
    dom_rate_by_type.sort_values('pct', ascending=True).tail(12)
)
overall_dom_pct = df['domestic'].mean() * 100
col_dom = [
    DOM_CLR if v >= overall_dom_pct else NDOM_CLR
    for v in dom_rate_by_type['pct']
]
bars = axes[0, 1].barh(dom_rate_by_type.index, dom_rate_by_type['pct'],
                       color=col_dom, edgecolor='white', alpha=0.92)
axes[0, 1].axvline(overall_dom_pct, color=DARK, linestyle='--',
                   linewidth=1.5, alpha=0.7,
                   label=f'Overall {overall_dom_pct:.1f}%')
axes[0, 1].set_title('Domestic Rate by Crime Type (%)\n(Top 12 by domestic rate)')
axes[0, 1].set_xlabel('Domestic Rate (%)')
axes[0, 1].legend(fontsize=8)
for bar, val in zip(bars, dom_rate_by_type['pct']):
    axes[0, 1].text(
        bar.get_width() + 0.3,
        bar.get_y() + bar.get_height() / 2,
        f'{val:.1f}%', va='center', fontsize=8
    )
axes[0, 1].tick_params(axis='y', labelsize=7.5)

# ── 6c: Domestic rate by hour ─────────────────────────────────
dom_by_hour = df.groupby('Hour')['domestic'].mean() * 100
axes[1, 0].plot(dom_by_hour.index, dom_by_hour.values,
                color=DOM_CLR, linewidth=2.5, marker='o',
                markersize=5, zorder=3)
axes[1, 0].fill_between(dom_by_hour.index, dom_by_hour.values,
                        alpha=0.20, color=DOM_CLR)
axes[1, 0].axhline(overall_dom_pct, color=DARK, linestyle='--',
                   linewidth=1.5, alpha=0.6,
                   label=f'Overall {overall_dom_pct:.1f}%')
axes[1, 0].set_title('Domestic Rate by Hour of Day')
axes[1, 0].set_xlabel('Hour (0=midnight, 12=noon)')
axes[1, 0].set_ylabel('Domestic Rate (%)')
axes[1, 0].legend()
axes[1, 0].set_xlim(0, 23)

# ── 6d: Domestic rate by day of week ─────────────────────────
dom_by_day = df.groupby('Day_of_Week')['domestic'].mean() * 100
axes[1, 1].bar(
    [DAY_NAMES[int(i)] for i in dom_by_day.index], dom_by_day.values,
    color=[
        DOM_CLR if v >= overall_dom_pct else NDOM_CLR
        for v in dom_by_day.values
    ],
    edgecolor='white', alpha=0.92
)
axes[1, 1].axhline(overall_dom_pct, color=DARK, linestyle='--',
                   linewidth=1.5, alpha=0.6,
                   label=f'Overall {overall_dom_pct:.1f}%')
axes[1, 1].set_title('Domestic Rate by Day of Week')
axes[1, 1].set_xlabel('Day')
axes[1, 1].set_ylabel('Domestic Rate (%)')
axes[1, 1].legend()

plt.tight_layout(pad=2.5)
save_fig('02_domestic_breakdown.png')


# ================================================================
# CHART 7: CRIME SEVERITY DISTRIBUTION
# ================================================================
print("\n[Chart 7] Crime Severity Distribution...")

# Define severity bands for analysis
SEV_BANDS = {
    'Low (1-3)':    (1, 3),
    'Medium (4-6)': (4, 6),
    'High (7-10)':  (7, 10),
}
df['severity_band'] = pd.cut(
    df['Crime_Severity_Score'],
    bins=[0, 3, 6, 10],
    labels=['Low (1-3)', 'Medium (4-6)', 'High (7-10)']
)
SEV_COLORS = {
    'Low (1-3)':    AC[4],
    'Medium (4-6)': AC[1],
    'High (7-10)':  NARR_CLR,
}

fig, axes = plt.subplots(2, 2, figsize=(18, 13))
fig.suptitle('Crime Severity Analysis', y=0.98)

# ── 7a: Severity score histogram ─────────────────────────────
score_counts = df['Crime_Severity_Score'].value_counts().sort_index()
bar_colors_sev = []
for score in score_counts.index:
    if score <= 3:
        bar_colors_sev.append(AC[4])
    elif score <= 6:
        bar_colors_sev.append(AC[1])
    else:
        bar_colors_sev.append(NARR_CLR)

bars = axes[0, 0].bar(score_counts.index, score_counts.values,
                      color=bar_colors_sev, edgecolor='white', alpha=0.92)
axes[0, 0].set_title('Crime Severity Score Distribution\n(1=Low, 10=Homicide)')
axes[0, 0].set_xlabel('Severity Score')
axes[0, 0].set_ylabel('Number of Crimes')
axes[0, 0].set_xticks(range(1, 11))
for bar, val in zip(bars, score_counts.values):
    axes[0, 0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + total * 0.003,
        f'{val:,}', ha='center', va='bottom', fontsize=7.5
    )
legend_patches = [
    mpatches.Patch(color=AC[4],    label='Low (1-3)'),
    mpatches.Patch(color=AC[1],    label='Medium (4-6)'),
    mpatches.Patch(color=NARR_CLR, label='High (7-10)'),
]
axes[0, 0].legend(handles=legend_patches, fontsize=9)

# ── 7b: Crimes by severity band ───────────────────────────────
band_counts = df['severity_band'].value_counts()
band_order  = ['Low (1-3)', 'Medium (4-6)', 'High (7-10)']
band_counts = band_counts.reindex(band_order, fill_value=0)

bars = axes[0, 1].bar(
    band_counts.index, band_counts.values,
    color=[SEV_COLORS[b] for b in band_order],
    edgecolor='white', alpha=0.92, width=0.5
)
axes[0, 1].set_title('Crimes by Severity Band')
axes[0, 1].set_ylabel('Number of Crimes')
for bar, val in zip(bars, band_counts.values):
    pct = val / total * 100
    axes[0, 1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + total * 0.003,
        f'{val:,}\n({pct:.1f}%)',
        ha='center', va='bottom', fontsize=10, fontweight='bold', color=DARK
    )
axes[0, 1].set_ylim(0, band_counts.max() * 1.22)

# ── 7c: Arrest rate by severity band ─────────────────────────
arr_by_sev = (
    df.groupby('severity_band', observed=True)['arrest'].mean() * 100
).reindex(band_order, fill_value=0)

bars = axes[1, 0].bar(
    arr_by_sev.index, arr_by_sev.values,
    color=[SEV_COLORS[b] for b in band_order],
    edgecolor='white', alpha=0.92, width=0.5
)
axes[1, 0].axhline(overall_arr_pct, color=DARK, linestyle='--',
                   linewidth=1.5, alpha=0.6,
                   label=f'Overall {overall_arr_pct:.1f}%')
axes[1, 0].set_title('Arrest Rate by Severity Band')
axes[1, 0].set_ylabel('Arrest Rate (%)')
axes[1, 0].legend(fontsize=8)
for bar, val in zip(bars, arr_by_sev.values):
    axes[1, 0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f'{val:.1f}%', ha='center', va='bottom',
        fontsize=11, fontweight='bold'
    )

# ── 7d: KDE of hour by severity band (scipy kde_fill) ─────────
# High-severity crimes: do they peak at different hours than low-severity?
for band, color in SEV_COLORS.items():
    sub = df[df['severity_band'] == band]['Hour'].dropna()
    if len(sub) < 10:
        continue
    kde_fill(
        axes[1, 1], sub, color,
        label=f'{band} (n={len(sub):,})',
        clip_lo=0, clip_hi=23,
        linestyle='-' if 'High' in band else '--',
        alpha_fill=0.20 if 'High' in band else 0.12,
        lw=2.5 if 'High' in band else 2.0
    )

axes[1, 1].set_title('Hour Distribution by Severity Band\n(KDE — scipy gaussian_kde)')
axes[1, 1].set_xlabel('Hour of Day')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_xlim(0, 23)
axes[1, 1].legend(fontsize=8)

# Clean up temp column
df.drop(columns=['severity_band'], inplace=True, errors='ignore')

plt.tight_layout(pad=2.5)
save_fig('02_severity_distribution.png')


# ================================================================
# FINAL SUMMARY
# ================================================================
_sep = "=" * 65
print(f"""
{_sep}
   EDA v1 COMPLETE — All 7 charts saved
{_sep}

PALETTE USED:
  Binary comparisons   : Dark2_r  (orange-red vs teal-green)
  Crime type bars      : tab20    (20 distinct colors)
  Category comparisons : Paired_r (multi-hue)
  Histograms           : Accent_r (one color per feature)
  Heatmaps             : YlOrRd   (yellow=low, red=high risk)

KDE METHOD: scipy gaussian_kde (direct x/y arrays)
  - No seaborn internal line reference bugs
  - All density fills clipped to valid x range

SAVED TO: docs/figures/
  02_crime_type_distribution.png
  02_geographic_scatter.png
  02_hourly_heatmap.png
  02_monthly_trend.png
  02_arrest_rates.png
  02_domestic_breakdown.png
  02_severity_distribution.png

KEY FINDINGS:
  Total crimes     : {total:,}
  Crime types      : {df['primary_type'].nunique()}
  Arrest rate      : {df['arrest'].mean()*100:.1f}%
  Domestic rate    : {df['domestic'].mean()*100:.1f}%
  Peak hour        : {int(df.groupby('Hour').size().idxmax()):02d}:00
  Peak season      : {df['Season'].value_counts().idxmax() if 'Season' in df.columns else 'N/A'}

NEXT: Section 4 — Feature Engineering (notebooks/03_feature_engineering.py)
""")
