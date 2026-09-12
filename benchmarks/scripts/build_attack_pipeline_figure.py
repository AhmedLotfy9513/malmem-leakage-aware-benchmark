# ============================================================
# Stage VIII-E2 — Revised Publication Figure
# Evidence-Aware ATT&CK Alignment Pipeline
#
# Improvements:
# - Two-row pipeline for readability
# - Explicit support equation S_k(x)
# - Separate candidate evidence panel
# - Publication-friendly layout
# - No ATT&CK detection claim
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = r"D:\lotfy\XPE_Memory_Project"

TABLE_PATH = os.path.join(
    PROJECT_DIR,
    "outputs",
    "attack_alignment",
    "stage_viii_e1",
    "manuscript_attack_evidence_table.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "outputs",
    "attack_alignment",
    "stage_viii_e2_revised"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. LOAD RESULTS
# ============================================================

df = pd.read_csv(TABLE_PATH)

required_columns = {
    "ATTACK_ID",
    "ATTACK_Name",
    "Evidence_Level",
    "Mapped_Feature_Count",
    "Malware_Support_Present_%",
    "Benign_Support_Present_%",
    "Rank_Biserial_Effect"
}

missing = required_columns - set(df.columns)

if missing:
    raise RuntimeError(
        "Missing required columns: "
        + str(sorted(missing))
    )


print("=" * 100)
print("STAGE VIII-E2 REVISED")
print("BUILDING PUBLICATION-QUALITY ATT&CK PIPELINE")
print("=" * 100)


# ============================================================
# 3. HELPERS
# ============================================================

def add_box(
    ax,
    x,
    y,
    w,
    h,
    title,
    body,
    title_size=10,
    body_size=8.5
):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.05",
        linewidth=1.3,
        fill=False
    )

    ax.add_patch(box)

    ax.text(
        x + w / 2,
        y + h * 0.67,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold"
    )

    ax.text(
        x + w / 2,
        y + h * 0.34,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        wrap=True
    )


def add_arrow(
    ax,
    x1,
    y1,
    x2,
    y2
):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="->",
        mutation_scale=14,
        linewidth=1.2
    )

    ax.add_patch(arrow)


# ============================================================
# 4. MAIN PIPELINE FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(16, 10)
)

ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis("off")


# ------------------------------------------------------------
# Title
# ------------------------------------------------------------

ax.text(
    8,
    9.5,
    "Evidence-Aware ATT&CK Alignment for Memory Malware Analysis",
    ha="center",
    va="center",
    fontsize=18,
    fontweight="bold"
)

ax.text(
    8,
    9.1,
    "Forensic evidence is translated into behavioral hypotheses before ATT&CK-aligned attribution support is quantified.",
    ha="center",
    va="center",
    fontsize=10
)


# ============================================================
# 5. ROW 1 — FORENSIC INTERPRETATION
# ============================================================

row1_y = 6.9
w = 3.1
h = 1.45

x1 = 0.4
x2 = 4.25
x3 = 8.1
x4 = 11.95


add_box(
    ax,
    x1,
    row1_y,
    w,
    h,
    "1. Memory Features",
    "55 numerical memory-forensics features\nfrom CIC-MalMem-2022"
)

add_box(
    ax,
    x2,
    row1_y,
    w,
    h,
    "2. Forensic Evidence",
    "Evidence sources include malfind,\nldrmodules, psxview, and contextual features"
)

add_box(
    ax,
    x3,
    row1_y,
    w,
    h,
    "3. Behavioral Hypothesis",
    "Observed forensic patterns are interpreted\nas behaviorally compatible evidence"
)

add_box(
    ax,
    x4,
    row1_y,
    w,
    h,
    "4. ATT&CK Candidate",
    "Candidate alignment only:\nT1055, T1620, and T1014"
)


add_arrow(
    ax,
    x1 + w,
    row1_y + h / 2,
    x2,
    row1_y + h / 2
)

add_arrow(
    ax,
    x2 + w,
    row1_y + h / 2,
    x3,
    row1_y + h / 2
)

add_arrow(
    ax,
    x3 + w,
    row1_y + h / 2,
    x4,
    row1_y + h / 2
)


# ============================================================
# 6. TRANSITION ARROW
# ============================================================

add_arrow(
    ax,
    x4 + w / 2,
    row1_y,
    x4 + w / 2,
    6.0
)


# ============================================================
# 7. ROW 2 — ATTRIBUTION AND VALIDATION
# ============================================================

row2_y = 4.2

add_box(
    ax,
    x4,
    row2_y,
    w,
    h,
    "5. Evidence Level",
    "HIGH_CANDIDATE, MEDIUM_CANDIDATE,\nor LOW_CONTEXTUAL"
)

add_box(
    ax,
    x3,
    row2_y,
    w,
    h,
    "6. Predicted-Class SHAP",
    "Top-15 local SHAP contributions\nfor the model-predicted family"
)

add_box(
    ax,
    x2,
    row2_y,
    w,
    h,
    "7. Candidate Support",
    "Positive SHAP contribution is aggregated\nfor reviewed ATT&CK-aligned features"
)

add_box(
    ax,
    x1,
    row2_y,
    w,
    h,
    "8. Statistical Characterization",
    "Mann–Whitney U, Holm correction,\nrank-biserial effect size, bootstrap CI"
)


add_arrow(
    ax,
    x4,
    row2_y + h / 2,
    x3 + w,
    row2_y + h / 2
)

add_arrow(
    ax,
    x3,
    row2_y + h / 2,
    x2 + w,
    row2_y + h / 2
)

add_arrow(
    ax,
    x2,
    row2_y + h / 2,
    x1 + w,
    row2_y + h / 2
)


# ============================================================
# 8. SUPPORT EQUATION
# ============================================================
# ============================================================
# 8. SUPPORT EQUATION
# ============================================================

equation = (
    r"$S_k(x)="
    r"\frac{\sum_{i\in\mathcal{F}_k}\max(\phi_i^{(\hat{y})}(x),0)}"
    r"{\sum_{j\in\mathrm{TopK}}\max(\phi_j^{(\hat{y})}(x),0)}$"
)

ax.text(
    8,
    3.38,
    equation,
    ha="center",
    va="center",
    fontsize=18
)

ax.text(
    8,
    2.70,
    (
        r"$K=15$; "
        r"$\phi_i^{(\hat{y})}(x)$ is the local SHAP value for the predicted class; "
        r"$\mathcal{F}_k$ is the set of reviewed features aligned with candidate $k$."
    ),
    ha="center",
    va="center",
    fontsize=9.2
)

# ============================================================
# 9. CANDIDATE SUMMARY PANEL
# ============================================================

panel_y = 0.95
panel_w = 4.6
panel_h = 1.35

candidate_positions = {
    "T1055": 0.45,
    "T1620": 5.7,
    "T1014": 10.95
}

for candidate_id, x in candidate_positions.items():

    row = df[
        df["ATTACK_ID"] == candidate_id
    ].iloc[0]

    title = (
        f"{row['ATTACK_ID']} — "
        f"{row['ATTACK_Name']}"
    )

    body = (
        f"{row['Evidence_Level']} | "
        f"{int(row['Mapped_Feature_Count'])} mapped features\n"
        f"Malware support: {row['Malware_Support_Present_%']:.1f}% | "
        f"Benign support: {row['Benign_Support_Present_%']:.1f}% | "
        f"$r_{{rb}}$={row['Rank_Biserial_Effect']:.3f}"
    )

    add_box(
        ax,
        x,
        panel_y,
        panel_w,
        panel_h,
        title,
        body,
        title_size=10,
        body_size=8.5
    )


# ============================================================
# 10. GUARDRAIL
# ============================================================

ax.text(
    8,
    0.25,
    (
        "Interpretation guardrail: ATT&CK alignment denotes evidence-supported behavioral hypotheses, "
        "not confirmed technique detection, causal attribution, or ground-truth ATT&CK labeling."
    ),
    ha="center",
    va="center",
    fontsize=9,
    fontstyle="italic"
)


plt.tight_layout()


pipeline_png = os.path.join(
    OUTPUT_DIR,
    "attack_alignment_pipeline_revised.png"
)

pipeline_pdf = os.path.join(
    OUTPUT_DIR,
    "attack_alignment_pipeline_revised.pdf"
)


plt.savefig(
    pipeline_png,
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    pipeline_pdf,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 11. REVISED COMPARISON FIGURE
# ============================================================

order = [
    "T1055",
    "T1620",
    "T1014"
]

plot_df = (
    df[
        [
            "ATTACK_ID",
            "Malware_Support_Present_%",
            "Benign_Support_Present_%"
        ]
    ]
    .copy()
)

plot_df["_order"] = (
    plot_df["ATTACK_ID"]
    .map(
        {
            cid: i
            for i, cid in enumerate(order)
        }
    )
)

plot_df = (
    plot_df
    .sort_values("_order")
    .drop(columns="_order")
)


x = list(
    range(
        len(plot_df)
    )
)

width = 0.34


fig, ax = plt.subplots(
    figsize=(8.5, 5.8)
)


bars_malware = ax.bar(
    [i - width / 2 for i in x],
    plot_df["Malware_Support_Present_%"],
    width,
    label="Malware"
)

bars_benign = ax.bar(
    [i + width / 2 for i in x],
    plot_df["Benign_Support_Present_%"],
    width,
    label="Benign"
)


ax.set_title(
    "Positive ATT&CK-Aligned Support Presence",
    fontsize=14,
    fontweight="bold"
)

ax.set_ylabel(
    "Samples with Positive Candidate Support (%)"
)

ax.set_xlabel(
    "ATT&CK Candidate"
)

ax.set_xticks(x)

ax.set_xticklabels(
    plot_df["ATTACK_ID"]
)

ax.set_ylim(
    0,
    105
)

ax.legend()


for bars in [
    bars_malware,
    bars_benign
]:
    for bar in bars:

        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1.3,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )


ax.text(
    0.5,
    -0.17,
    (
        "Positive support denotes model attribution associated with reviewed forensic evidence; "
        "values are not ATT&CK technique-detection rates."
    ),
    transform=ax.transAxes,
    ha="center",
    va="center",
    fontsize=8.3,
    fontstyle="italic"
)


plt.tight_layout()


comparison_png = os.path.join(
    OUTPUT_DIR,
    "attack_candidate_support_comparison_revised.png"
)

comparison_pdf = os.path.join(
    OUTPUT_DIR,
    "attack_candidate_support_comparison_revised.pdf"
)


plt.savefig(
    comparison_png,
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    comparison_pdf,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 12. CAPTIONS
# ============================================================

caption_path = os.path.join(
    OUTPUT_DIR,
    "revised_figure_captions.txt"
)


with open(
    caption_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "FIGURE 1 CAPTION\n"
        "Evidence-aware ATT&CK alignment pipeline. "
        "Memory-forensics features are first interpreted as forensic observations "
        "and behavioral hypotheses before alignment with candidate ATT&CK techniques. "
        "Feature-level evidence strength constrains the mapping, while predicted-class "
        "local SHAP values quantify positive model attribution associated with each "
        "candidate. Candidate support is normalized over positive Top-15 SHAP contributions "
        "and subsequently characterized using non-parametric statistical testing and effect sizes. "
        "The alignment layer does not constitute confirmed ATT&CK technique detection.\n\n"
    )

    f.write(
        "FIGURE 2 CAPTION\n"
        "Presence of positive ATT&CK-aligned SHAP support in the family-balanced analytical cohort. "
        "T1055, T1620, and T1014 indicate candidate behavioral alignment only. "
        "The percentages represent cases in which at least one reviewed candidate-aligned feature "
        "made a positive predicted-class SHAP contribution."
    )


# ============================================================
# 13. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 100)
print("FILES SAVED")
print("=" * 100)

print("1.", pipeline_png)
print("2.", pipeline_pdf)
print("3.", comparison_png)
print("4.", comparison_pdf)
print("5.", caption_path)

print("\nStage VIII-E2 revised complete.")