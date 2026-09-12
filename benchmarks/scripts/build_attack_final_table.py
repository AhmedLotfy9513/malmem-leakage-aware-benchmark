# ============================================================
# Stage VIII-E1
# Final ATT&CK Evidence Table for Manuscript
#
# PURPOSE
# ------------------------------------------------------------
# Build a publication-ready summary table for the reviewed
# ATT&CK-aligned behavioral hypotheses.
#
# The table combines:
#   - ATT&CK candidate ID and name
#   - Evidence level
#   - Reviewed forensic features
#   - Malware/Benign support prevalence
#   - Malware/Benign median normalized SHAP support
#   - Rank-biserial effect size
#   - Holm-adjusted p-value
#   - Interpretation
#
# IMPORTANT
# ------------------------------------------------------------
# This table reports ATT&CK-aligned behavioral hypotheses.
# It does NOT report confirmed ATT&CK technique detections.
# ============================================================

import os
import json
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = r"D:\lotfy\XPE_Memory_Project"

D2_DIR = os.path.join(
    PROJECT_DIR,
    "outputs",
    "attack_alignment",
    "stage_viii_d2"
)

MAPPING_PATH = os.path.join(
    PROJECT_DIR,
    "outputs",
    "attack_alignment",
    "attack_feature_evidence_mapping_v2.csv"
)

STATS_PATH = os.path.join(
    D2_DIR,
    "attack_support_statistical_tests.csv"
)

PREVALENCE_PATH = os.path.join(
    D2_DIR,
    "attack_candidate_prevalence.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "outputs",
    "attack_alignment",
    "stage_viii_e1"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. LOAD INPUTS
# ============================================================

print("=" * 100)
print("STAGE VIII-E1")
print("FINAL ATT&CK EVIDENCE TABLE")
print("=" * 100)


mapping_df = pd.read_csv(
    MAPPING_PATH
)

stats_df = pd.read_csv(
    STATS_PATH
)

prevalence_df = pd.read_csv(
    PREVALENCE_PATH
)


print(
    f"\nMapping rows    : {len(mapping_df)}"
)

print(
    f"Statistical rows: {len(stats_df)}"
)

print(
    f"Prevalence rows : {len(prevalence_df)}"
)


# ============================================================
# 3. NORMALIZE TEXT COLUMNS
# ============================================================

mapping_text_cols = [
    "feature_name",
    "attack_candidate_id",
    "attack_candidate_name",
    "evidence_level",
    "forensic_observation",
    "behavioral_hypothesis"
]


for col in mapping_text_cols:

    if col in mapping_df.columns:

        mapping_df[col] = (
            mapping_df[col]
            .fillna("")
            .astype(str)
        )


# ============================================================
# 4. KEEP ATT&CK-CANDIDATE FEATURES ONLY
# ============================================================

candidate_mapping_df = mapping_df[
    mapping_df[
        "attack_candidate_id"
    ].str.strip()
    !=
    ""
].copy()


candidate_ids = sorted(
    candidate_mapping_df[
        "attack_candidate_id"
    ]
    .unique()
    .tolist()
)


print(
    "\nReviewed ATT&CK candidates:",
    candidate_ids
)


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def get_prevalence_row(
    candidate_id,
    cohort
):

    rows = prevalence_df[
        (
            prevalence_df[
                "ATTACK_Candidate_ID"
            ]
            ==
            candidate_id
        )
        &
        (
            prevalence_df[
                "Cohort"
            ]
            ==
            cohort
        )
    ]


    if len(rows) != 1:

        raise RuntimeError(
            f"Expected one prevalence row for "
            f"{candidate_id} / {cohort}, found {len(rows)}"
        )


    return rows.iloc[0]


def get_stats_row(
    candidate_id
):

    pattern = (
        f"{candidate_id} support: Malware vs Benign"
    )


    rows = stats_df[
        stats_df[
            "Comparison"
        ]
        ==
        pattern
    ]


    if len(rows) != 1:

        raise RuntimeError(
            f"Expected one statistical row for "
            f"{candidate_id}, found {len(rows)}"
        )


    return rows.iloc[0]


def evidence_strength_label(
    levels
):

    levels = set(
        levels
    )


    if "HIGH_CANDIDATE" in levels:

        return "HIGH_CANDIDATE"

    if "MEDIUM_CANDIDATE" in levels:

        return "MEDIUM_CANDIDATE"

    return "LOW_CONTEXTUAL"


def effect_interpretation(
    effect
):

    value = abs(
        float(
            effect
        )
    )


    if value < 0.10:
        return "Negligible separation"

    if value < 0.30:
        return "Small separation"

    if value < 0.50:
        return "Moderate separation"

    return "Large separation"


# ============================================================
# 6. BUILD FINAL TABLE
# ============================================================

final_rows = []


for candidate_id in candidate_ids:

    candidate_features = candidate_mapping_df[
        candidate_mapping_df[
            "attack_candidate_id"
        ]
        ==
        candidate_id
    ].copy()


    candidate_name_values = (
        candidate_features[
            "attack_candidate_name"
        ]
        .dropna()
        .astype(str)
        .str.strip()
    )


    candidate_name_values = [
        value
        for value in candidate_name_values
        if value != ""
    ]


    if candidate_name_values:

        candidate_name = (
            candidate_name_values[0]
        )

    else:

        candidate_name = (
            "Unnamed ATT&CK candidate"
        )


    feature_names = (
        candidate_features[
            "feature_name"
        ]
        .astype(str)
        .tolist()
    )


    evidence_level = (
        evidence_strength_label(
            candidate_features[
                "evidence_level"
            ]
            .astype(str)
            .tolist()
        )
    )


    malware_prev = get_prevalence_row(
        candidate_id,
        "MALWARE_TRUE"
    )


    benign_prev = get_prevalence_row(
        candidate_id,
        "BENIGN_TRUE"
    )


    stats_row = get_stats_row(
        candidate_id
    )


    effect = float(
        stats_row[
            "Rank_Biserial_Effect"
        ]
    )


    holm_p = float(
        stats_row[
            "Holm_Adjusted_P_Value"
        ]
    )


    # --------------------------------------------------------
    # Manuscript-safe interpretation
    # --------------------------------------------------------

    if candidate_id == "T1055":

        interpretation = (
            "High-evidence process-injection-aligned forensic "
            "support was observed in a substantial subset of "
            "malware cases and was absent from benign cases in "
            "the analyzed cohort. This remains an ATT&CK-aligned "
            "behavioral hypothesis rather than confirmed technique "
            "detection."
        )

    elif candidate_id == "T1620":

        interpretation = (
            "Reflective-code-loading-aligned support showed strong "
            "distributional separation between malware and benign "
            "samples. Because aligned loader anomalies also occurred "
            "in benign cases, the evidence is interpreted as "
            "behaviorally compatible rather than technique-confirming."
        )

    elif candidate_id == "T1014":

        interpretation = (
            "Rootkit-aligned process-concealment evidence occurred "
            "more frequently in malware than benign samples, with "
            "moderate distributional separation. The evidence remains "
            "indirect and requires analyst validation."
        )

    else:

        interpretation = (
            "The reviewed forensic evidence showed statistical "
            "separation between malware and benign samples, but "
            "is interpreted only as an ATT&CK-aligned behavioral "
            "hypothesis."
        )


    final_rows.append(
        {
            "ATTACK_ID":
                candidate_id,

            "ATTACK_Name":
                candidate_name,

            "Evidence_Level":
                evidence_level,

            "Mapped_Feature_Count":
                int(
                    len(
                        feature_names
                    )
                ),

            "Mapped_Features":
                "; ".join(
                    feature_names
                ),

            "Malware_Support_Present_Rate":
                float(
                    malware_prev[
                        "Support_Present_Rate"
                    ]
                ),

            "Benign_Support_Present_Rate":
                float(
                    benign_prev[
                        "Support_Present_Rate"
                    ]
                ),

            "Malware_Median_Support":
                float(
                    stats_row[
                        "Median_A"
                    ]
                ),

            "Benign_Median_Support":
                float(
                    stats_row[
                        "Median_B"
                    ]
                ),

            "Rank_Biserial_Effect":
                effect,

            "Effect_Interpretation":
                effect_interpretation(
                    effect
                ),

            "Holm_Adjusted_P":
                holm_p,

            "Statistically_Significant":
                bool(
                    stats_row[
                        "Significant_After_Holm_0_05"
                    ]
                ),

            "Interpretation":
                interpretation,

            "Direct_Detection_Claim":
                False,

            "Causal_Claim":
                False,

            "Analyst_Validation_Required":
                True
        }
    )


final_df = pd.DataFrame(
    final_rows
)


# ============================================================
# 7. ORDER TABLE
# ============================================================

preferred_order = [
    "T1055",
    "T1620",
    "T1014"
]


order_map = {
    candidate_id: idx
    for idx, candidate_id
    in enumerate(
        preferred_order
    )
}


final_df[
    "_order"
] = (
    final_df[
        "ATTACK_ID"
    ]
    .map(
        order_map
    )
    .fillna(
        999
    )
)


final_df = (
    final_df
    .sort_values(
        [
            "_order",
            "ATTACK_ID"
        ]
    )
    .drop(
        columns=[
            "_order"
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 8. CREATE COMPACT MANUSCRIPT TABLE
# ============================================================

compact_df = final_df[
    [
        "ATTACK_ID",
        "ATTACK_Name",
        "Evidence_Level",
        "Mapped_Feature_Count",
        "Malware_Support_Present_Rate",
        "Benign_Support_Present_Rate",
        "Malware_Median_Support",
        "Benign_Median_Support",
        "Rank_Biserial_Effect",
        "Effect_Interpretation",
        "Holm_Adjusted_P"
    ]
].copy()


compact_df[
    "Malware_Support_Present_%"
] = (
    100
    *
    compact_df[
        "Malware_Support_Present_Rate"
    ]
)


compact_df[
    "Benign_Support_Present_%"
] = (
    100
    *
    compact_df[
        "Benign_Support_Present_Rate"
    ]
)


compact_df = compact_df[
    [
        "ATTACK_ID",
        "ATTACK_Name",
        "Evidence_Level",
        "Mapped_Feature_Count",
        "Malware_Support_Present_%",
        "Benign_Support_Present_%",
        "Malware_Median_Support",
        "Benign_Median_Support",
        "Rank_Biserial_Effect",
        "Effect_Interpretation",
        "Holm_Adjusted_P"
    ]
]


# ============================================================
# 9. GENERATE LATEX TABLE
# ============================================================

latex_path = os.path.join(
    OUTPUT_DIR,
    "attack_evidence_table.tex"
)


latex_lines = []

latex_lines.append(
    r"\begin{table*}[t]"
)

latex_lines.append(
    r"\centering"
)

latex_lines.append(
    r"\caption{Evidence-aware ATT\&CK alignment derived from local SHAP attribution. "
    r"Candidate support denotes attribution associated with reviewed forensic evidence "
    r"and must not be interpreted as confirmed ATT\&CK technique detection.}"
)

latex_lines.append(
    r"\label{tab:attack_evidence_alignment}"
)

latex_lines.append(
    r"\resizebox{\textwidth}{!}{%"
)

latex_lines.append(
    r"\begin{tabular}{llcccccccc}"
)

latex_lines.append(
    r"\hline"
)

latex_lines.append(
    r"ATT\&CK & Candidate & Evidence & Features & Malware (\%) & Benign (\%) & "
    r"Med. Malware & Med. Benign & $r_{rb}$ & Holm-$p$ \\"
)

latex_lines.append(
    r"\hline"
)


for _, row in compact_df.iterrows():

    latex_lines.append(

        f"{row['ATTACK_ID']} & "
        f"{row['ATTACK_Name']} & "
        f"{row['Evidence_Level']} & "
        f"{int(row['Mapped_Feature_Count'])} & "
        f"{row['Malware_Support_Present_%']:.2f} & "
        f"{row['Benign_Support_Present_%']:.2f} & "
        f"{row['Malware_Median_Support']:.4f} & "
        f"{row['Benign_Median_Support']:.4f} & "
        f"{row['Rank_Biserial_Effect']:.3f} & "
        f"{row['Holm_Adjusted_P']:.2e} "
        r"\\"
    )


latex_lines.append(
    r"\hline"
)

latex_lines.append(
    r"\end{tabular}%"
)

latex_lines.append(
    r"}"
)

latex_lines.append(
    r"\end{table*}"
)


with open(
    latex_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(
            latex_lines
        )
    )


# ============================================================
# 10. SAVE CSV OUTPUTS
# ============================================================

full_path = os.path.join(
    OUTPUT_DIR,
    "final_attack_evidence_table.csv"
)


compact_path = os.path.join(
    OUTPUT_DIR,
    "manuscript_attack_evidence_table.csv"
)


final_df.to_csv(
    full_path,
    index=False
)


compact_df.to_csv(
    compact_path,
    index=False
)


# ============================================================
# 11. SAVE SUMMARY JSON
# ============================================================

summary = {

    "stage":
        "VIII-E1",

    "candidate_count":
        int(
            len(
                final_df
            )
        ),

    "candidate_ids":
        final_df[
            "ATTACK_ID"
        ]
        .tolist(),

    "interpretation_policy":
        (
            "ATT&CK-aligned behavioral hypotheses; "
            "not confirmed ATT&CK technique detections"
        ),

    "direct_detection_claim":
        False,

    "causal_claim":
        False,

    "analyst_validation_required":
        True
}


summary_path = os.path.join(
    OUTPUT_DIR,
    "stage_viii_e1_summary.json"
)


with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# 12. FINAL DISPLAY
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "FINAL MANUSCRIPT TABLE"
)

print(
    "=" * 100
)


print(
    compact_df.to_string(
        index=False
    )
)


print(
    "\n" + "=" * 100
)

print(
    "FILES SAVED"
)

print(
    "=" * 100
)


print(
    "1.",
    full_path
)

print(
    "2.",
    compact_path
)

print(
    "3.",
    latex_path
)

print(
    "4.",
    summary_path
)


print(
    "\nStage VIII-E1 complete."
)