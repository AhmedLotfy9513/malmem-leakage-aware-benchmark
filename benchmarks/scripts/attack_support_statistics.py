# ============================================================
# Stage VIII-D2
# Statistical Characterization of ATT&CK-Aligned SHAP Support
#
# PURPOSE
# ------------------------------------------------------------
# 1) Statistically compare ATT&CK-aligned SHAP support:
#       - Malware vs Benign
#       - Errors vs Correct predictions
#
# 2) Characterize individual ATT&CK candidates:
#       - T1014
#       - T1055
#       - T1620
#       - or any candidates present in the reviewed rule base
#
# 3) Report:
#       - Mann-Whitney U
#       - Holm-adjusted p-values
#       - Rank-biserial effect size
#       - Common-language probability
#       - Bootstrap 95% CI for median difference
#
# IMPORTANT SCIENTIFIC GUARDRAILS
# ------------------------------------------------------------
# - ATT&CK support is NOT a detection probability.
# - ATT&CK candidates are behavioral hypotheses.
# - SHAP explains model behavior, not causality.
# - Missing candidate support is represented as ZERO.
#   This prevents candidate-presence conditioning bias.
# ============================================================

import os
import json
import warnings

import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu


warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = r"D:\lotfy\XPE_Memory_Project"

D1_DIR = os.path.join(
    PROJECT_DIR,
    "outputs",
    "attack_alignment",
    "stage_viii_d1"
)

CASE_PATH = os.path.join(
    D1_DIR,
    "batch_attack_case_summary.csv"
)

CANDIDATE_PATH = os.path.join(
    D1_DIR,
    "batch_attack_candidate_support.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "outputs",
    "attack_alignment",
    "stage_viii_d2"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


RANDOM_STATE = 42

N_BOOTSTRAP = 5000

ALPHA = 0.05

EPS = 1e-12


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def holm_adjust(p_values):
    """
    Holm-Bonferroni family-wise error correction.

    Returns adjusted p-values in original order.
    """

    p_values = np.asarray(
        p_values,
        dtype=float
    )

    m = len(
        p_values
    )

    order = np.argsort(
        p_values
    )

    ordered_p = p_values[
        order
    ]

    adjusted_ordered = np.zeros(
        m,
        dtype=float
    )


    running_max = 0.0


    for rank, p in enumerate(
        ordered_p
    ):

        multiplier = (
            m
            -
            rank
        )

        adjusted = min(
            1.0,
            multiplier * p
        )

        running_max = max(
            running_max,
            adjusted
        )

        adjusted_ordered[
            rank
        ] = running_max


    adjusted = np.empty(
        m,
        dtype=float
    )

    adjusted[
        order
    ] = adjusted_ordered


    return adjusted


def bootstrap_median_difference(
    group_a,
    group_b,
    n_bootstrap=N_BOOTSTRAP,
    random_state=RANDOM_STATE
):
    """
    Bootstrap CI for:

        median(group_a) - median(group_b)

    Positive value means Group A has higher median support.
    """

    a = np.asarray(
        group_a,
        dtype=float
    )

    b = np.asarray(
        group_b,
        dtype=float
    )


    rng = np.random.default_rng(
        random_state
    )


    diffs = np.empty(
        n_bootstrap,
        dtype=float
    )


    for i in range(
        n_bootstrap
    ):

        boot_a = rng.choice(
            a,
            size=len(a),
            replace=True
        )

        boot_b = rng.choice(
            b,
            size=len(b),
            replace=True
        )


        diffs[i] = (
            np.median(
                boot_a
            )
            -
            np.median(
                boot_b
            )
        )


    lower = float(
        np.percentile(
            diffs,
            2.5
        )
    )

    upper = float(
        np.percentile(
            diffs,
            97.5
        )
    )


    return lower, upper


def compare_groups(
    group_a,
    group_b,
    comparison_name,
    group_a_name,
    group_b_name,
    random_state=RANDOM_STATE
):
    """
    Mann-Whitney comparison.

    Direction:
        positive effect size =>
        Group A tends to have HIGHER values than Group B.

    Common-language probability:
        estimated P(A > B), with ties contributing implicitly
        through the Mann-Whitney ranking framework.
    """

    a = np.asarray(
        group_a,
        dtype=float
    )

    b = np.asarray(
        group_b,
        dtype=float
    )


    if len(a) == 0 or len(b) == 0:

        raise ValueError(
            f"Empty group in comparison: {comparison_name}"
        )


    u_stat, p_value = mannwhitneyu(
        a,
        b,
        alternative="two-sided",
        method="auto"
    )


    denominator = (
        len(a)
        *
        len(b)
    )


    common_language = float(
        u_stat
        /
        denominator
    )


    rank_biserial = float(
        2.0
        *
        common_language
        -
        1.0
    )


    median_a = float(
        np.median(
            a
        )
    )

    median_b = float(
        np.median(
            b
        )
    )


    mean_a = float(
        np.mean(
            a
        )
    )

    mean_b = float(
        np.mean(
            b
        )
    )


    median_difference = (
        median_a
        -
        median_b
    )


    ci_low, ci_high = (
        bootstrap_median_difference(
            a,
            b,
            random_state=random_state
        )
    )


    return {

        "Comparison":
            comparison_name,

        "Group_A":
            group_a_name,

        "Group_B":
            group_b_name,

        "N_A":
            int(
                len(a)
            ),

        "N_B":
            int(
                len(b)
            ),

        "Mean_A":
            mean_a,

        "Mean_B":
            mean_b,

        "Median_A":
            median_a,

        "Median_B":
            median_b,

        "Median_Difference_A_minus_B":
            float(
                median_difference
            ),

        "Median_Difference_CI95_Low":
            ci_low,

        "Median_Difference_CI95_High":
            ci_high,

        "Mann_Whitney_U":
            float(
                u_stat
            ),

        "Raw_P_Value":
            float(
                p_value
            ),

        "Rank_Biserial_Effect":
            rank_biserial,

        "Common_Language_P_A_gt_B":
            common_language
    }


# ============================================================
# 3. LOAD STAGE VIII-D1 OUTPUTS
# ============================================================

print(
    "=" * 100
)

print(
    "STAGE VIII-D2"
)

print(
    "STATISTICAL CHARACTERIZATION OF ATT&CK-ALIGNED SHAP SUPPORT"
)

print(
    "=" * 100
)


case_df = pd.read_csv(
    CASE_PATH
)


candidate_df = pd.read_csv(
    CANDIDATE_PATH
)


print(
    f"\nCase rows      : {len(case_df)}"
)

print(
    f"Candidate rows : {len(candidate_df)}"
)


# ============================================================
# 4. VALIDATE INPUT
# ============================================================

required_case_columns = {

    "Cohort_Position",
    "True_Family",
    "Predicted_Family",
    "Prediction_Correct",
    "ATTACK_Normalized_SHAP_Support"
}


missing_case_columns = (
    required_case_columns
    -
    set(
        case_df.columns
    )
)


if missing_case_columns:

    raise RuntimeError(
        "Missing case columns: "
        +
        str(
            sorted(
                missing_case_columns
            )
        )
    )


required_candidate_columns = {

    "Cohort_Position",
    "ATTACK_Candidate_ID",
    "Candidate_Normalized_SHAP_Support"
}


missing_candidate_columns = (
    required_candidate_columns
    -
    set(
        candidate_df.columns
    )
)


if missing_candidate_columns:

    raise RuntimeError(
        "Missing candidate columns: "
        +
        str(
            sorted(
                missing_candidate_columns
            )
        )
    )


if case_df[
    "Cohort_Position"
].duplicated().any():

    raise RuntimeError(
        "Duplicate Cohort_Position values in case summary."
    )


if len(
    case_df
) != 2000:

    print(
        "WARNING: Expected analytical cohort of 2000, "
        f"but found {len(case_df)}."
    )


# ============================================================
# 5. NORMALIZE BOOLEAN COLUMN
# ============================================================

if (
    case_df[
        "Prediction_Correct"
    ].dtype
    !=
    bool
):

    case_df[
        "Prediction_Correct"
    ] = (
        case_df[
            "Prediction_Correct"
        ]
        .astype(str)
        .str.lower()
        .map(
            {
                "true": True,
                "false": False
            }
        )
    )


if case_df[
    "Prediction_Correct"
].isna().any():

    raise RuntimeError(
        "Could not normalize Prediction_Correct."
    )


# ============================================================
# 6. GLOBAL COMPARISONS
# ============================================================

print(
    "\nRunning global statistical comparisons..."
)


statistical_rows = []


# ------------------------------------------------------------
# A. MALWARE vs BENIGN
#
# Direction:
#     Group A = Malware
#     Group B = Benign
#
# Positive effect => malware support is higher.
# ------------------------------------------------------------

malware_values = (
    case_df[
        case_df[
            "True_Family"
        ]
        !=
        "Benign"
    ][
        "ATTACK_Normalized_SHAP_Support"
    ]
    .to_numpy(
        dtype=float
    )
)


benign_values = (
    case_df[
        case_df[
            "True_Family"
        ]
        ==
        "Benign"
    ][
        "ATTACK_Normalized_SHAP_Support"
    ]
    .to_numpy(
        dtype=float
    )
)


statistical_rows.append(

    compare_groups(
        malware_values,
        benign_values,
        comparison_name=(
            "Overall ATT&CK support: Malware vs Benign"
        ),
        group_a_name="Malware",
        group_b_name="Benign",
        random_state=42
    )
)


# ------------------------------------------------------------
# B. ERROR vs CORRECT
#
# Direction:
#     Group A = Error
#     Group B = Correct
#
# Positive effect => error cases have higher support.
# ------------------------------------------------------------

error_values = (
    case_df[
        case_df[
            "Prediction_Correct"
        ]
        ==
        False
    ][
        "ATTACK_Normalized_SHAP_Support"
    ]
    .to_numpy(
        dtype=float
    )
)


correct_values = (
    case_df[
        case_df[
            "Prediction_Correct"
        ]
        ==
        True
    ][
        "ATTACK_Normalized_SHAP_Support"
    ]
    .to_numpy(
        dtype=float
    )
)


statistical_rows.append(

    compare_groups(
        error_values,
        correct_values,
        comparison_name=(
            "Overall ATT&CK support: Error vs Correct"
        ),
        group_a_name="Error",
        group_b_name="Correct",
        random_state=43
    )
)


# ============================================================
# 7. BUILD ZERO-INCLUSIVE CANDIDATE MATRIX
# ============================================================

print(
    "Building zero-inclusive ATT&CK candidate matrix..."
)


candidate_ids = sorted(

    candidate_df[
        "ATTACK_Candidate_ID"
    ]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


candidate_ids = [

    candidate

    for candidate in candidate_ids

    if candidate.strip()
    !=
    ""
]


print(
    "Candidates found:",
    candidate_ids
)


candidate_pivot = (
    candidate_df

    .pivot_table(
        index="Cohort_Position",
        columns="ATTACK_Candidate_ID",
        values="Candidate_Normalized_SHAP_Support",
        aggfunc="sum",
        fill_value=0.0
    )

    .reindex(
        case_df[
            "Cohort_Position"
        ],
        fill_value=0.0
    )
)


candidate_pivot.index.name = (
    "Cohort_Position"
)


candidate_pivot = (
    candidate_pivot
    .reset_index()
)


candidate_matrix_df = (
    case_df[
        [
            "Cohort_Position",
            "True_Family",
            "Predicted_Family",
            "Prediction_Correct"
        ]
    ]

    .merge(
        candidate_pivot,
        on="Cohort_Position",
        how="left",
        validate="one_to_one"
    )
)


for candidate in candidate_ids:

    candidate_matrix_df[
        candidate
    ] = (
        candidate_matrix_df[
            candidate
        ]
        .fillna(
            0.0
        )
        .astype(float)
    )


print(
    "[PASS] Missing candidate appearances represented as zero support."
)


# ============================================================
# 8. CANDIDATE-SPECIFIC MALWARE vs BENIGN
# ============================================================

print(
    "\nRunning candidate-specific comparisons..."
)


for candidate_index, candidate_id in enumerate(
    candidate_ids
):

    malware_candidate = (
        candidate_matrix_df[
            candidate_matrix_df[
                "True_Family"
            ]
            !=
            "Benign"
        ][
            candidate_id
        ]
        .to_numpy(
            dtype=float
        )
    )


    benign_candidate = (
        candidate_matrix_df[
            candidate_matrix_df[
                "True_Family"
            ]
            ==
            "Benign"
        ][
            candidate_id
        ]
        .to_numpy(
            dtype=float
        )
    )


    result = compare_groups(

        malware_candidate,

        benign_candidate,

        comparison_name=(
            f"{candidate_id} support: Malware vs Benign"
        ),

        group_a_name="Malware",

        group_b_name="Benign",

        random_state=(
            100
            +
            candidate_index
        )
    )


    result[
        "ATTACK_Candidate_ID"
    ] = candidate_id


    statistical_rows.append(
        result
    )


# ============================================================
# 9. CREATE STATISTICAL RESULTS TABLE
# ============================================================

stats_df = pd.DataFrame(
    statistical_rows
)


# ============================================================
# 10. HOLM MULTIPLE-TESTING CORRECTION
# ============================================================

adjusted_p_values = holm_adjust(

    stats_df[
        "Raw_P_Value"
    ]
    .to_numpy(
        dtype=float
    )
)


stats_df[
    "Holm_Adjusted_P_Value"
] = adjusted_p_values


stats_df[
    "Significant_After_Holm_0_05"
] = (
    stats_df[
        "Holm_Adjusted_P_Value"
    ]
    <
    ALPHA
)


# ============================================================
# 11. EFFECT DIRECTION
# ============================================================

def effect_direction(row):

    effect = float(
        row[
            "Rank_Biserial_Effect"
        ]
    )


    if abs(
        effect
    ) < EPS:

        return "NO_DIRECTION"


    if effect > 0:

        return (
            f"{row['Group_A']}_HIGHER"
        )

    return (
        f"{row['Group_B']}_HIGHER"
    )


stats_df[
    "Effect_Direction"
] = stats_df.apply(
    effect_direction,
    axis=1
)


# ============================================================
# 12. DESCRIPTIVE EFFECT MAGNITUDE
# ============================================================
#
# These labels are descriptive only.
# They are NOT detection thresholds.
#
# Absolute rank-biserial:
#   < 0.10    negligible
#   < 0.30    small
#   < 0.50    moderate
#   >= 0.50   large
# ============================================================

def descriptive_effect_magnitude(
    effect
):

    value = abs(
        float(
            effect
        )
    )


    if value < 0.10:

        return "NEGLIGIBLE"

    elif value < 0.30:

        return "SMALL"

    elif value < 0.50:

        return "MODERATE"

    else:

        return "LARGE"


stats_df[
    "Effect_Magnitude_Descriptive"
] = (

    stats_df[
        "Rank_Biserial_Effect"
    ]

    .apply(
        descriptive_effect_magnitude
    )
)


# ============================================================
# 13. CANDIDATE PREVALENCE
# ============================================================

prevalence_rows = []


for candidate_id in candidate_ids:

    for cohort_name, subset in {

        "ALL":
            candidate_matrix_df,

        "BENIGN_TRUE":
            candidate_matrix_df[
                candidate_matrix_df[
                    "True_Family"
                ]
                ==
                "Benign"
            ],

        "MALWARE_TRUE":
            candidate_matrix_df[
                candidate_matrix_df[
                    "True_Family"
                ]
                !=
                "Benign"
            ]

    }.items():


        values = (
            subset[
                candidate_id
            ]
            .to_numpy(
                dtype=float
            )
        )


        present = (
            values
            >
            0
        )


        prevalence_rows.append(
            {
                "ATTACK_Candidate_ID":
                    candidate_id,

                "Cohort":
                    cohort_name,

                "N":
                    int(
                        len(
                            values
                        )
                    ),

                "Support_Present_N":
                    int(
                        np.sum(
                            present
                        )
                    ),

                "Support_Present_Rate":
                    float(
                        np.mean(
                            present
                        )
                    ),

                "Mean_Support_Including_Zero":
                    float(
                        np.mean(
                            values
                        )
                    ),

                "Median_Support_Including_Zero":
                    float(
                        np.median(
                            values
                        )
                    ),

                "P75_Support_Including_Zero":
                    float(
                        np.percentile(
                            values,
                            75
                        )
                    ),

                "P90_Support_Including_Zero":
                    float(
                        np.percentile(
                            values,
                            90
                        )
                    )
            }
        )


prevalence_df = pd.DataFrame(
    prevalence_rows
)


# ============================================================
# 14. PER-FAMILY CANDIDATE SUPPORT
# ============================================================

family_candidate_rows = []


for family_name, subset in (

    candidate_matrix_df

    .groupby(
        "True_Family"
    )
):

    for candidate_id in candidate_ids:

        values = (
            subset[
                candidate_id
            ]
            .to_numpy(
                dtype=float
            )
        )


        family_candidate_rows.append(
            {
                "True_Family":
                    family_name,

                "ATTACK_Candidate_ID":
                    candidate_id,

                "N":
                    int(
                        len(
                            values
                        )
                    ),

                "Mean_Support":
                    float(
                        np.mean(
                            values
                        )
                    ),

                "Median_Support":
                    float(
                        np.median(
                            values
                        )
                    ),

                "Support_Present_Rate":
                    float(
                        np.mean(
                            values
                            >
                            0
                        )
                    ),

                "P75_Support":
                    float(
                        np.percentile(
                            values,
                            75
                        )
                    ),

                "P90_Support":
                    float(
                        np.percentile(
                            values,
                            90
                        )
                    )
            }
        )


family_candidate_df = pd.DataFrame(
    family_candidate_rows
)


# ============================================================
# 15. SCIENTIFIC SAFETY CHECKS
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "SCIENTIFIC SAFETY CHECKS"
)

print(
    "=" * 100
)


assert (
    len(
        candidate_matrix_df
    )
    ==
    len(
        case_df
    )
)


print(
    "[PASS] Candidate matrix preserves every cohort case."
)


for candidate_id in candidate_ids:

    assert (
        (
            candidate_matrix_df[
                candidate_id
            ]
            >=
            0.0
        ).all()
    )

    assert (
        (
            candidate_matrix_df[
                candidate_id
            ]
            <=
            1.0000001
        ).all()
    )


print(
    "[PASS] Candidate-specific normalized support remains within [0, 1]."
)


assert (
    (
        case_df[
            "ATTACK_Normalized_SHAP_Support"
        ]
        >=
        0
    ).all()
)


assert (
    (
        case_df[
            "ATTACK_Normalized_SHAP_Support"
        ]
        <=
        1.0000001
    ).all()
)


print(
    "[PASS] Overall ATT&CK support remains within [0, 1]."
)


assert (
    stats_df[
        "Holm_Adjusted_P_Value"
    ].between(
        0,
        1
    ).all()
)


print(
    "[PASS] Multiple-comparison adjusted p-values valid."
)


print(
    "[PASS] No ATT&CK detection or causal inference is performed."
)


# ============================================================
# 16. SAVE OUTPUTS
# ============================================================

stats_path = os.path.join(
    OUTPUT_DIR,
    "attack_support_statistical_tests.csv"
)


prevalence_path = os.path.join(
    OUTPUT_DIR,
    "attack_candidate_prevalence.csv"
)


candidate_matrix_path = os.path.join(
    OUTPUT_DIR,
    "attack_candidate_zero_inclusive_matrix.csv"
)


family_candidate_path = os.path.join(
    OUTPUT_DIR,
    "attack_candidate_support_by_family.csv"
)


summary_path = os.path.join(
    OUTPUT_DIR,
    "stage_viii_d2_summary.json"
)


stats_df.to_csv(
    stats_path,
    index=False
)


prevalence_df.to_csv(
    prevalence_path,
    index=False
)


candidate_matrix_df.to_csv(
    candidate_matrix_path,
    index=False
)


family_candidate_df.to_csv(
    family_candidate_path,
    index=False
)


# ============================================================
# 17. JSON SUMMARY
# ============================================================

malware_benign_row = stats_df[
    stats_df[
        "Comparison"
    ]
    ==
    "Overall ATT&CK support: Malware vs Benign"
].iloc[0]


error_correct_row = stats_df[
    stats_df[
        "Comparison"
    ]
    ==
    "Overall ATT&CK support: Error vs Correct"
].iloc[0]


summary = {

    "stage":
        "VIII-D2",

    "analysis":
        (
            "Statistical characterization of "
            "ATT&CK-aligned SHAP support"
        ),

    "cohort_size":
        int(
            len(
                case_df
            )
        ),

    "candidate_ids":
        candidate_ids,

    "multiple_testing":
        "Holm-Bonferroni",

    "alpha":
        ALPHA,

    "bootstrap_iterations":
        N_BOOTSTRAP,

    "overall_malware_vs_benign": {

        "median_difference":
            float(
                malware_benign_row[
                    "Median_Difference_A_minus_B"
                ]
            ),

        "rank_biserial_effect":
            float(
                malware_benign_row[
                    "Rank_Biserial_Effect"
                ]
            ),

        "holm_adjusted_p":
            float(
                malware_benign_row[
                    "Holm_Adjusted_P_Value"
                ]
            )
    },

    "overall_error_vs_correct": {

        "median_difference":
            float(
                error_correct_row[
                    "Median_Difference_A_minus_B"
                ]
            ),

        "rank_biserial_effect":
            float(
                error_correct_row[
                    "Rank_Biserial_Effect"
                ]
            ),

        "holm_adjusted_p":
            float(
                error_correct_row[
                    "Holm_Adjusted_P_Value"
                ]
            )
    },

    "guardrails": {

        "attack_detection_claim":
            False,

        "causal_claim":
            False,

        "candidate_absence_encoded_as_zero":
            True,

        "support_interpreted_as_probability":
            False,

        "support_bands_created":
            False,

        "analyst_validation_required":
            True
    }
}


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
# 18. PRINT MAIN STATISTICAL RESULTS
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "STATISTICAL RESULTS"
)

print(
    "=" * 100
)


display_columns = [

    "Comparison",

    "N_A",

    "N_B",

    "Median_A",

    "Median_B",

    "Median_Difference_A_minus_B",

    "Median_Difference_CI95_Low",

    "Median_Difference_CI95_High",

    "Rank_Biserial_Effect",

    "Effect_Magnitude_Descriptive",

    "Raw_P_Value",

    "Holm_Adjusted_P_Value",

    "Significant_After_Holm_0_05"
]


print(

    stats_df[
        display_columns
    ]

    .to_string(
        index=False
    )
)


print(
    "\n" + "=" * 100
)

print(
    "ATT&CK CANDIDATE PREVALENCE"
)

print(
    "=" * 100
)


print(

    prevalence_df

    .to_string(
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
    stats_path
)

print(
    "2.",
    prevalence_path
)

print(
    "3.",
    candidate_matrix_path
)

print(
    "4.",
    family_candidate_path
)

print(
    "5.",
    summary_path
)


print(
    "\nStage VIII-D2 complete."
)


print(
    "\nINTERPRETATION GUARDRAIL:"
)

print(
    "Statistical separation in ATT&CK-aligned SHAP support "
    "does not convert the alignment layer into an ATT&CK "
    "technique detector. Results describe model-attribution "
    "patterns associated with reviewed forensic evidence."
)