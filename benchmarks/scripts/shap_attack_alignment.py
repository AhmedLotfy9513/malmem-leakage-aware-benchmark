# ============================================================
# Stage VIII-C
# Local SHAP -> Forensic Evidence -> MITRE ATT&CK Alignment
#
# IMPORTANT SCIENTIFIC PRINCIPLE
# --------------------------------
# This script does NOT perform ATT&CK detection.
#
# It performs:
#
# Model Prediction
#      ↓
# Local SHAP Explanation
#      ↓
# Forensic Feature Interpretation
#      ↓
# Evidence-Level ATT&CK Candidate Alignment
#
# ATT&CK candidates are supporting behavioral hypotheses only.
#
# Author workflow:
# CIC-MalMem-2022 family classifier
# + Local SHAP
# + Feature-level forensic evidence rules
# ============================================================

import os
import json
import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = r"D:\lotfy\XPE_Memory_Project"

SHAP_FILE = os.path.join(
    PROJECT_DIR,
    "outputs",
    "shap_deep_analysis",
    "local_shap_contributions.csv",
)

CASE_FILE = os.path.join(
    PROJECT_DIR,
    "outputs",
    "shap_multiclass_analysis",
    "local_explanation_cases.csv",
)

MAPPING_FILE = os.path.join(
    PROJECT_DIR,
    "outputs",
    "attack_alignment",
    "attack_feature_evidence_mapping_v2.csv",
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "outputs",
    "attack_alignment",
    "stage_viii_c",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

TOP_K = 15

# We use the explanation of the model prediction
# for ATT&CK-support analysis.
PRIMARY_EXPLANATION_TYPE = "Predicted_Class"


# ============================================================
# 3. LOAD INPUT DATA
# ============================================================

print("=" * 88)
print("STAGE VIII-C")
print("LOCAL SHAP -> FORENSIC EVIDENCE -> ATT&CK CANDIDATE ALIGNMENT")
print("=" * 88)


shap_df = pd.read_csv(
    SHAP_FILE
)

case_df = pd.read_csv(
    CASE_FILE
)

mapping_df = pd.read_csv(
    MAPPING_FILE
)


print("\nLoaded:")

print(
    f"  SHAP contribution rows : {len(shap_df)}"
)

print(
    f"  Explanation cases      : {len(case_df)}"
)

print(
    f"  Evidence rules         : {len(mapping_df)}"
)


# ============================================================
# 4. VALIDATE REQUIRED COLUMNS
# ============================================================

required_shap_columns = {
    "Case",
    "Sample_Index",
    "True_Family",
    "Predicted_Family",
    "Explained_Class",
    "Explanation_Type",
    "Rank",
    "Feature",
    "Feature_Value",
    "SHAP_Value",
    "Absolute_SHAP",
}


required_case_columns = {
    "Case",
    "Sample_Index",
    "True_Family",
    "Predicted_Family",
    "Confidence",
}


required_mapping_columns = {
    "feature_name",
    "forensic_source",
    "forensic_observation",
    "behavioral_hypothesis",
    "attack_candidate_id",
    "attack_candidate_name",
    "evidence_level",
    "direct_detection_allowed",
    "causal_claim_allowed",
    "analyst_validation_required",
    "note",
}


missing_shap = (
    required_shap_columns
    -
    set(shap_df.columns)
)

missing_case = (
    required_case_columns
    -
    set(case_df.columns)
)

missing_mapping = (
    required_mapping_columns
    -
    set(mapping_df.columns)
)


if missing_shap:

    raise RuntimeError(
        "Missing SHAP columns: "
        + str(sorted(missing_shap))
    )


if missing_case:

    raise RuntimeError(
        "Missing case columns: "
        + str(sorted(missing_case))
    )


if missing_mapping:

    raise RuntimeError(
        "Missing ATT&CK mapping columns: "
        + str(sorted(missing_mapping))
    )


# ============================================================
# 5. NORMALIZE TEXT / NaN VALUES
# ============================================================

text_columns_mapping = [
    "attack_candidate_id",
    "attack_candidate_name",
    "evidence_level",
    "forensic_source",
    "forensic_observation",
    "behavioral_hypothesis",
    "note",
]


for col in text_columns_mapping:

    mapping_df[col] = (
        mapping_df[col]
        .fillna("")
        .astype(str)
    )


# ============================================================
# 6. VERIFY EXACT FEATURE COVERAGE
# ============================================================

mapping_features = set(
    mapping_df["feature_name"]
)

shap_features = set(
    shap_df["Feature"]
)


unmapped_features = (
    shap_features
    -
    mapping_features
)


if unmapped_features:

    raise RuntimeError(
        "SHAP contains features without reviewed "
        "forensic mapping:\n"
        + "\n".join(
            sorted(unmapped_features)
        )
    )


print(
    "\n[PASS] Every SHAP feature has a reviewed "
    "feature-level evidence rule."
)


# ============================================================
# 7. KEEP PREDICTED-CLASS EXPLANATIONS
#
# Why?
# We want to explain which evidence influenced the MODEL'S
# actual prediction.
#
# True-class explanations remain useful for later error
# analysis but are not mixed into prediction support.
# ============================================================

primary_df = shap_df[
    shap_df["Explanation_Type"]
    ==
    PRIMARY_EXPLANATION_TYPE
].copy()


primary_df = primary_df[
    primary_df["Rank"]
    <=
    TOP_K
].copy()


if len(primary_df) == 0:

    raise RuntimeError(
        "No Predicted_Class SHAP rows found."
    )


print(
    f"[PASS] Using Top-{TOP_K} local SHAP features "
    "for predicted-class explanations."
)


# ============================================================
# 8. MERGE SHAP WITH FORENSIC RULE BASE
# ============================================================

merged_df = primary_df.merge(
    mapping_df,
    left_on="Feature",
    right_on="feature_name",
    how="left",
    validate="many_to_one",
)


if merged_df[
    "feature_name"
].isna().any():

    raise RuntimeError(
        "Evidence-rule merge produced missing mappings."
    )


# ============================================================
# 9. ADD CASE CONFIDENCE
# ============================================================

case_meta = (
    case_df[
        [
            "Sample_Index",
            "Confidence",
        ]
    ]
    .drop_duplicates(
        subset=[
            "Sample_Index"
        ]
    )
)


merged_df = merged_df.merge(
    case_meta,
    on="Sample_Index",
    how="left",
    validate="many_to_one",
)


# ============================================================
# 10. SHAP DIRECTION
#
# Positive SHAP:
#     Supports the explained class.
#
# Negative SHAP:
#     Opposes the explained class.
#
# Absolute SHAP:
#     Importance magnitude only.
# ============================================================

merged_df[
    "SHAP_Direction"
] = np.where(
    merged_df["SHAP_Value"] > 0,
    "SUPPORTS_EXPLAINED_CLASS",
    np.where(
        merged_df["SHAP_Value"] < 0,
        "OPPOSES_EXPLAINED_CLASS",
        "NEUTRAL",
    ),
)


merged_df[
    "Positive_SHAP_Support"
] = np.maximum(
    merged_df["SHAP_Value"],
    0.0,
)


# ============================================================
# 11. ATT&CK CANDIDATE FLAGS
# ============================================================

merged_df[
    "Has_ATTACK_Candidate"
] = (
    merged_df[
        "attack_candidate_id"
    ]
    .fillna("")
    .str.strip()
    .ne("")
)


merged_df[
    "Supports_ATTACK_Candidate"
] = (
    merged_df[
        "Has_ATTACK_Candidate"
    ]
    &
    (
        merged_df[
            "SHAP_Value"
        ]
        >
        0
    )
)


# ============================================================
# 12. PER-SAMPLE TOTAL POSITIVE SHAP
# ============================================================

positive_totals = (
    merged_df
    .groupby(
        "Sample_Index"
    )[
        "Positive_SHAP_Support"
    ]
    .sum()
    .rename(
        "Total_Positive_SHAP"
    )
)


merged_df = merged_df.merge(
    positive_totals,
    on="Sample_Index",
    how="left",
)


# ============================================================
# 13. NORMALIZED LOCAL SHAP SUPPORT
#
# Contribution of each positively supporting feature relative
# to all positive Top-K SHAP support for that prediction.
# ============================================================

merged_df[
    "Normalized_Positive_SHAP_Support"
] = np.where(
    merged_df[
        "Total_Positive_SHAP"
    ]
    >
    0,

    merged_df[
        "Positive_SHAP_Support"
    ]
    /
    merged_df[
        "Total_Positive_SHAP"
    ],

    0.0,
)


# ============================================================
# 14. FEATURE-LEVEL EVIDENCE TABLE
# ============================================================

feature_output_columns = [

    "Case",
    "Sample_Index",

    "True_Family",
    "Predicted_Family",
    "Confidence",

    "Rank",

    "Feature",
    "Feature_Value",

    "SHAP_Value",
    "Absolute_SHAP",
    "SHAP_Direction",

    "Positive_SHAP_Support",
    "Normalized_Positive_SHAP_Support",

    "forensic_source",
    "forensic_observation",
    "behavioral_hypothesis",

    "attack_candidate_id",
    "attack_candidate_name",
    "evidence_level",

    "Has_ATTACK_Candidate",
    "Supports_ATTACK_Candidate",

    "direct_detection_allowed",
    "causal_claim_allowed",
    "analyst_validation_required",

    "note",
]


feature_alignment_df = (
    merged_df[
        feature_output_columns
    ]
    .sort_values(
        [
            "Sample_Index",
            "Rank",
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 15. BUILD ATT&CK SUPPORT TABLE
#
# Only:
#   - reviewed ATT&CK candidates
#   - positive SHAP support
#
# are allowed to contribute to support scores.
# ============================================================

supported_attack_df = merged_df[
    merged_df[
        "Supports_ATTACK_Candidate"
    ]
].copy()


attack_support_rows = []


if len(
    supported_attack_df
) > 0:

    grouped = (
        supported_attack_df
        .groupby(
            [
                "Case",
                "Sample_Index",
                "True_Family",
                "Predicted_Family",
                "attack_candidate_id",
                "attack_candidate_name",
            ],
            dropna=False,
        )
    )


    for keys, group in grouped:

        (
            case_name,
            sample_index,
            true_family,
            predicted_family,
            attack_id,
            attack_name,
        ) = keys


        positive_shap = float(
            group[
                "Positive_SHAP_Support"
            ].sum()
        )


        total_positive_shap = float(
            group[
                "Total_Positive_SHAP"
            ].iloc[0]
        )


        normalized_support = (
            positive_shap
            /
            total_positive_shap
            if total_positive_shap > 0
            else 0.0
        )


        evidence_levels = set(
            group[
                "evidence_level"
            ]
        )


        # Conservative evidence aggregation:
        #
        # HIGH only if at least one HIGH feature contributes.
        # Otherwise MEDIUM.
        if (
            "HIGH_CANDIDATE"
            in
            evidence_levels
        ):

            aggregated_level = (
                "HIGH_CANDIDATE"
            )

        elif (
            "MEDIUM_CANDIDATE"
            in
            evidence_levels
        ):

            aggregated_level = (
                "MEDIUM_CANDIDATE"
            )

        else:

            aggregated_level = (
                "LOW_CONTEXTUAL"
            )


        supporting_features = (
            group[
                "Feature"
            ]
            .astype(str)
            .tolist()
        )


        supporting_feature_text = (
            "; ".join(
                supporting_features
            )
        )


        attack_support_rows.append(
            {
                "Case":
                    case_name,

                "Sample_Index":
                    int(sample_index),

                "True_Family":
                    true_family,

                "Predicted_Family":
                    predicted_family,

                "ATTACK_Candidate_ID":
                    attack_id,

                "ATTACK_Candidate_Name":
                    attack_name,

                "Aggregated_Evidence_Level":
                    aggregated_level,

                "Supporting_Feature_Count":
                    len(
                        supporting_features
                    ),

                "Supporting_Features":
                    supporting_feature_text,

                "Candidate_Positive_SHAP":
                    positive_shap,

                "Candidate_Normalized_SHAP_Support":
                    normalized_support,

                "Direct_Detection_Claim":
                    False,

                "ATTACK_Ground_Truth":
                    False,

                "Analyst_Validation_Required":
                    True,
            }
        )


attack_support_df = pd.DataFrame(
    attack_support_rows
)


# ============================================================
# 16. CASE-LEVEL SUMMARY
# ============================================================

case_summary_rows = []


for sample_index, group in (
    merged_df.groupby(
        "Sample_Index"
    )
):

    first = group.iloc[0]


    positive_attack_rows = group[
        group[
            "Supports_ATTACK_Candidate"
        ]
    ]


    candidate_ids = sorted(
        set(
            positive_attack_rows[
                "attack_candidate_id"
            ]
            .dropna()
            .astype(str)
        )
        -
        {""}
    )


    candidate_names = sorted(
        set(
            positive_attack_rows[
                "attack_candidate_name"
            ]
            .dropna()
            .astype(str)
        )
        -
        {""}
    )


    attack_support = float(
        positive_attack_rows[
            "Positive_SHAP_Support"
        ].sum()
    )


    total_positive = float(
        group[
            "Positive_SHAP_Support"
        ].sum()
    )


    normalized_attack_support = (
        attack_support
        /
        total_positive
        if total_positive > 0
        else 0.0
    )


    high_count = int(
        (
            positive_attack_rows[
                "evidence_level"
            ]
            ==
            "HIGH_CANDIDATE"
        ).sum()
    )


    medium_count = int(
        (
            positive_attack_rows[
                "evidence_level"
            ]
            ==
            "MEDIUM_CANDIDATE"
        ).sum()
    )


    contextual_count = int(
        (
            group[
                "evidence_level"
            ]
            ==
            "LOW_CONTEXTUAL"
        ).sum()
    )


    if high_count > 0:

        highest_evidence = (
            "HIGH_CANDIDATE"
        )

    elif medium_count > 0:

        highest_evidence = (
            "MEDIUM_CANDIDATE"
        )

    else:

        highest_evidence = (
            "CONTEXT_ONLY"
        )


    if len(candidate_ids) == 0:

        analyst_statement = (
            "No ATT&CK candidate is supported by "
            "the positively contributing Top-K SHAP "
            "features. The explanation remains "
            "forensic-contextual only."
        )

    else:

        analyst_statement = (
            "One or more ATT&CK-aligned behavioral "
            "hypotheses are supported by positively "
            "contributing local SHAP features. "
            "These are candidate alignments only, "
            "not technique detections."
        )


    case_summary_rows.append(
        {
            "Case":
                first["Case"],

            "Sample_Index":
                int(sample_index),

            "True_Family":
                first[
                    "True_Family"
                ],

            "Predicted_Family":
                first[
                    "Predicted_Family"
                ],

            "Prediction_Correct":
                (
                    first[
                        "True_Family"
                    ]
                    ==
                    first[
                        "Predicted_Family"
                    ]
                ),

            "Confidence":
                first[
                    "Confidence"
                ],

            "Top_K":
                TOP_K,

            "ATTACK_Candidate_Count":
                len(
                    candidate_ids
                ),

            "ATTACK_Candidate_IDs":
                "; ".join(
                    candidate_ids
                ),

            "ATTACK_Candidate_Names":
                "; ".join(
                    candidate_names
                ),

            "Highest_Evidence_Level":
                highest_evidence,

            "High_Evidence_Feature_Count":
                high_count,

            "Medium_Evidence_Feature_Count":
                medium_count,

            "Contextual_Feature_Count":
                contextual_count,

            "Total_Positive_SHAP":
                total_positive,

            "ATTACK_Positive_SHAP":
                attack_support,

            "ATTACK_Normalized_SHAP_Support":
                normalized_attack_support,

            "Direct_Detection_Claim":
                False,

            "Causal_Claim":
                False,

            "ATTACK_Ground_Truth":
                False,

            "Analyst_Validation_Required":
                True,

            "Analyst_Statement":
                analyst_statement,
        }
    )


case_summary_df = pd.DataFrame(
    case_summary_rows
)


# ============================================================
# 17. SCIENTIFIC SAFETY CHECKS
# ============================================================

print("\n" + "=" * 88)
print("SCIENTIFIC SAFETY CHECKS")
print("=" * 88)


# ------------------------------------------------------------
# No LOW_CONTEXTUAL feature may produce a candidate ID
# ------------------------------------------------------------

low_with_candidate = mapping_df[
    (
        mapping_df[
            "evidence_level"
        ]
        ==
        "LOW_CONTEXTUAL"
    )
    &
    (
        mapping_df[
            "attack_candidate_id"
        ]
        .fillna("")
        .str.strip()
        .ne("")
    )
]


assert (
    len(low_with_candidate)
    ==
    0
), (
    "Safety failure: LOW_CONTEXTUAL features "
    "must not generate ATT&CK candidates."
)


print(
    "[PASS] LOW_CONTEXTUAL features generate "
    "no ATT&CK candidate."
)


# ------------------------------------------------------------
# Negative SHAP cannot support ATT&CK candidate
# ------------------------------------------------------------

invalid_negative_support = merged_df[
    (
        merged_df[
            "SHAP_Value"
        ]
        <=
        0
    )
    &
    (
        merged_df[
            "Supports_ATTACK_Candidate"
        ]
    )
]


assert (
    len(
        invalid_negative_support
    )
    ==
    0
)


print(
    "[PASS] Negative SHAP values cannot support "
    "an ATT&CK candidate."
)


# ------------------------------------------------------------
# Direct detection disabled
# ------------------------------------------------------------

assert (
    mapping_df[
        "direct_detection_allowed"
    ]
    .astype(str)
    .str.lower()
    .isin(
        [
            "true",
            "1",
        ]
    )
    .sum()
    ==
    0
)


print(
    "[PASS] Direct ATT&CK detection remains disabled."
)


# ------------------------------------------------------------
# svcscan cannot generate candidate
# ------------------------------------------------------------

svcscan_candidates = merged_df[
    (
        merged_df[
            "forensic_source"
        ]
        ==
        "svcscan"
    )
    &
    (
        merged_df[
            "Has_ATTACK_Candidate"
        ]
    )
]


assert (
    len(
        svcscan_candidates
    )
    ==
    0
)


print(
    "[PASS] svcscan aggregate features generate "
    "no direct ATT&CK candidate."
)


# ============================================================
# 18. SAVE OUTPUT FILES
# ============================================================

feature_path = os.path.join(
    OUTPUT_DIR,
    "local_shap_forensic_alignment.csv",
)

attack_path = os.path.join(
    OUTPUT_DIR,
    "local_attack_candidate_support.csv",
)

case_summary_path = os.path.join(
    OUTPUT_DIR,
    "local_attack_case_summary.csv",
)

json_path = os.path.join(
    OUTPUT_DIR,
    "stage_viii_c_summary.json",
)


feature_alignment_df.to_csv(
    feature_path,
    index=False,
)


attack_support_df.to_csv(
    attack_path,
    index=False,
)


case_summary_df.to_csv(
    case_summary_path,
    index=False,
)


# ============================================================
# 19. JSON EXPERIMENT SUMMARY
# ============================================================

summary_json = {

    "stage":
        "VIII-C",

    "name":
        (
            "Local SHAP to Evidence-Aware "
            "MITRE ATT&CK Alignment"
        ),

    "top_k":
        TOP_K,

    "primary_explanation_type":
        PRIMARY_EXPLANATION_TYPE,

    "shap_support_policy": {
        "positive_shap":
            (
                "May support the explained prediction."
            ),

        "negative_shap":
            (
                "Cannot be counted as ATT&CK support."
            ),

        "absolute_shap":
            (
                "Used only as importance magnitude, "
                "not directional evidence."
            ),
    },

    "reasoning_chain": [
        "prediction",
        "local_shap_feature",
        "forensic_observation",
        "behavioral_hypothesis",
        "optional_attack_candidate",
        "evidence_level",
        "analyst_validation",
    ],

    "guardrails": {
        "attack_detection_claim":
            False,

        "causal_claim":
            False,

        "attack_ground_truth":
            False,

        "negative_shap_as_support":
            False,

        "analyst_validation_required":
            True,
    },

    "cases_analyzed":
        int(
            case_summary_df[
                "Sample_Index"
            ].nunique()
        ),

    "candidate_support_rows":
        int(
            len(
                attack_support_df
            )
        ),

    "candidate_cases":
        int(
            (
                case_summary_df[
                    "ATTACK_Candidate_Count"
                ]
                >
                0
            ).sum()
        ),

    "context_only_cases":
        int(
            (
                case_summary_df[
                    "ATTACK_Candidate_Count"
                ]
                ==
                0
            ).sum()
        ),
}


with open(
    json_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary_json,
        f,
        indent=4,
        ensure_ascii=False,
    )


# ============================================================
# 20. PRINT CASE RESULTS
# ============================================================

print("\n" + "=" * 88)
print("CASE-LEVEL ATT&CK ALIGNMENT")
print("=" * 88)


display_columns = [

    "Sample_Index",
    "True_Family",
    "Predicted_Family",
    "Prediction_Correct",
    "Confidence",
    "ATTACK_Candidate_Count",
    "ATTACK_Candidate_IDs",
    "Highest_Evidence_Level",
    "ATTACK_Normalized_SHAP_Support",
]


print(
    case_summary_df[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# 21. PRINT ATT&CK SUPPORTING FEATURES
# ============================================================

print("\n" + "=" * 88)
print("POSITIVELY SUPPORTING ATT&CK CANDIDATE FEATURES")
print("=" * 88)


if len(
    attack_support_df
) == 0:

    print(
        "No positively contributing ATT&CK "
        "candidate features were found."
    )

else:

    print(
        attack_support_df[
            [
                "Sample_Index",
                "Predicted_Family",
                "ATTACK_Candidate_ID",
                "ATTACK_Candidate_Name",
                "Aggregated_Evidence_Level",
                "Supporting_Features",
                "Candidate_Normalized_SHAP_Support",
            ]
        ]
        .to_string(
            index=False
        )
    )


# ============================================================
# 22. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 88)
print("FILES SAVED")
print("=" * 88)

print(
    "1.",
    feature_path
)

print(
    "2.",
    attack_path
)

print(
    "3.",
    case_summary_path
)

print(
    "4.",
    json_path
)


print("\nStage VIII-C complete.")

print(
    "\nIMPORTANT:"
)

print(
    "Reported ATT&CK entries are behavioral "
    "candidate alignments supported by local "
    "model explanations; they are NOT direct "
    "ATT&CK detections or ground-truth labels."
)