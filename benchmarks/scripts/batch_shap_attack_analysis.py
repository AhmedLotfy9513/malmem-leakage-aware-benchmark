# ============================================================
# Stage VIII-D1
# Batch Local SHAP + Evidence-Aware ATT&CK Support Analysis
#
# PURPOSE
# ------------------------------------------------------------
# 1) Reproduce the SAME group-aware 16-class holdout protocol.
# 2) Recreate the SAME Flat XGBoost configuration.
# 3) Build the SAME stratified SHAP cohort (~2000 samples).
# 4) Compute SHAP for the predicted class for each sample.
# 5) Extract Top-15 local features.
# 6) Join them with reviewed feature-level ATT&CK evidence rules.
# 7) Quantify ATT&CK-aligned positive SHAP support.
#
# IMPORTANT
# ------------------------------------------------------------
# ATT&CK candidate alignment != ATT&CK detection.
# SHAP explains model behavior, not malware causality.
# ============================================================

import os
import json
import warnings

import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score

from xgboost import XGBClassifier


warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = r"D:\lotfy\XPE_Memory_Project"

DATA_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "MalMem_LockedSplit.csv"
)

MAPPING_PATH = os.path.join(
    PROJECT_DIR,
    "outputs",
    "attack_alignment",
    "attack_feature_evidence_mapping_v2.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "outputs",
    "attack_alignment",
    "stage_viii_d1"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


RANDOM_STATE = 42

N_SHAP_SAMPLES = 2000

TOP_K = 15


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 96)
print("STAGE VIII-D1")
print("BATCH SHAP + ATT&CK SUPPORT ANALYSIS")
print("=" * 96)


df = pd.read_csv(
    DATA_PATH
)


dev = df[
    df["Split"] == "Development"
].copy()


print(
    f"\nDevelopment rows: {len(dev)}"
)


# ============================================================
# 3. FAMILY LABEL FUNCTION
# ============================================================

def get_family(category):

    category = str(category)

    if category == "Benign":
        return "Benign"

    parts = category.split("-")

    if len(parts) < 2:

        raise ValueError(
            f"Unexpected Category value: {category}"
        )

    return parts[1]


dev["Family_Label"] = (
    dev["Category"]
    .apply(get_family)
)


# ============================================================
# 4. FEATURE COLUMNS
# ============================================================

NON_FEATURE_COLUMNS = {
    "Category",
    "Class",
    "Split",
    "Group_ID",
    "Family_Label"
}


feature_columns = [

    col

    for col in dev.columns

    if (
        col not in NON_FEATURE_COLUMNS

        and

        pd.api.types.is_numeric_dtype(
            dev[col]
        )
    )
]


if len(feature_columns) != 55:

    raise RuntimeError(
        f"Expected 55 numeric features, found {len(feature_columns)}."
    )


print(
    f"Feature count: {len(feature_columns)}"
)


X = (
    dev[
        feature_columns
    ]
    .to_numpy(
        dtype=np.float32
    )
)


groups = (
    dev[
        "Group_ID"
    ]
    .astype(str)
    .to_numpy()
)


# ============================================================
# 5. FAMILY ENCODING
# ============================================================

family_names = sorted(
    dev[
        "Family_Label"
    ]
    .unique()
    .tolist()
)


if "Benign" in family_names:

    family_names.remove(
        "Benign"
    )

    family_names = (
        ["Benign"]
        +
        family_names
    )


family_to_id = {

    name: idx

    for idx, name

    in enumerate(
        family_names
    )
}


id_to_family = {

    idx: name

    for name, idx

    in family_to_id.items()
}


y = (
    dev[
        "Family_Label"
    ]
    .map(
        family_to_id
    )
    .to_numpy(
        dtype=int
    )
)


N_CLASSES = len(
    family_names
)


print(
    f"Family count: {N_CLASSES}"
)


# ============================================================
# 6. REPRODUCE SAME GROUP-AWARE HOLDOUT
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


train_idx, explain_idx = next(
    cv.split(
        X,
        y,
        groups
    )
)


train_groups = set(
    groups[
        train_idx
    ]
)

holdout_groups = set(
    groups[
        explain_idx
    ]
)


overlap = (
    train_groups
    &
    holdout_groups
)


if overlap:

    raise RuntimeError(
        "Group leakage detected between train and holdout."
    )


print(
    "[PASS] Group overlap = 0"
)


X_train = X[
    train_idx
]

y_train = y[
    train_idx
]

X_holdout = X[
    explain_idx
]

y_holdout = y[
    explain_idx
]


print(
    f"Train rows   : {len(X_train)}"
)

print(
    f"Holdout rows : {len(X_holdout)}"
)


# ============================================================
# 7. RECREATE SAME FLAT XGBOOST
# ============================================================

model = XGBClassifier(

    objective="multi:softprob",

    num_class=N_CLASSES,

    n_estimators=300,

    max_depth=6,

    learning_rate=0.05,

    subsample=0.9,

    colsample_bytree=0.9,

    eval_metric="mlogloss",

    tree_method="hist",

    random_state=RANDOM_STATE,

    n_jobs=-1
)


print(
    "\nTraining Flat XGBoost..."
)


model.fit(
    X_train,
    y_train
)


holdout_prob = model.predict_proba(
    X_holdout
)


holdout_pred = np.argmax(
    holdout_prob,
    axis=1
)


holdout_acc = accuracy_score(
    y_holdout,
    holdout_pred
)


holdout_macro_f1 = f1_score(
    y_holdout,
    holdout_pred,
    average="macro"
)


print(
    f"Holdout Accuracy : {holdout_acc:.6f}"
)

print(
    f"Holdout Macro-F1 : {holdout_macro_f1:.6f}"
)


# ============================================================
# 8. REPRODUCE SAME STRATIFIED SHAP COHORT
# ============================================================

rng = np.random.default_rng(
    RANDOM_STATE
)


selected_indices = []


per_class_target = max(
    1,
    N_SHAP_SAMPLES
    //
    N_CLASSES
)


for class_id in range(
    N_CLASSES
):

    cls_idx = np.where(
        y_holdout
        ==
        class_id
    )[0]


    take = min(
        len(cls_idx),
        per_class_target
    )


    if take > 0:

        chosen = rng.choice(
            cls_idx,
            size=take,
            replace=False
        )

        selected_indices.extend(
            chosen.tolist()
        )


selected_indices = np.asarray(
    selected_indices,
    dtype=int
)


X_shap = X_holdout[
    selected_indices
]

y_shap = y_holdout[
    selected_indices
]

pred_shap = holdout_pred[
    selected_indices
]

prob_shap = holdout_prob[
    selected_indices
]


print(
    f"\nSHAP cohort size: {len(X_shap)}"
)


# ============================================================
# 9. COHORT DISTRIBUTION
# ============================================================

cohort_distribution_rows = []


for class_id, family_name in enumerate(
    family_names
):

    count = int(
        np.sum(
            y_shap
            ==
            class_id
        )
    )

    cohort_distribution_rows.append(
        {
            "Family":
                family_name,

            "Count":
                count
        }
    )


cohort_distribution_df = pd.DataFrame(
    cohort_distribution_rows
)


print(
    "\nCohort distribution:"
)

print(
    cohort_distribution_df.to_string(
        index=False
    )
)


# ============================================================
# 10. BUILD SHAP EXPLAINER
# ============================================================

print(
    "\nBuilding TreeExplainer..."
)


explainer = shap.TreeExplainer(
    model
)


print(
    "Computing batch SHAP values..."
)


raw_shap = explainer.shap_values(
    X_shap
)


# ============================================================
# 11. NORMALIZE SHAP ARRAY SHAPE
# ============================================================

if isinstance(
    raw_shap,
    list
):

    shap_values = np.stack(
        raw_shap,
        axis=2
    )

else:

    shap_values = np.asarray(
        raw_shap
    )


if (
    shap_values.ndim == 3

    and

    shap_values.shape[1]
    ==
    len(feature_columns)

    and

    shap_values.shape[2]
    ==
    N_CLASSES
):

    pass


elif (
    shap_values.ndim == 3

    and

    shap_values.shape[1]
    ==
    N_CLASSES

    and

    shap_values.shape[2]
    ==
    len(feature_columns)
):

    shap_values = np.transpose(
        shap_values,
        (
            0,
            2,
            1
        )
    )


else:

    raise RuntimeError(
        "Unexpected SHAP array shape: "
        + str(
            shap_values.shape
        )
    )


print(
    f"[PASS] SHAP shape: {shap_values.shape}"
)


# ============================================================
# 12. LOAD FEATURE-LEVEL ATT&CK RULES
# ============================================================

mapping_df = pd.read_csv(
    MAPPING_PATH
)


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


missing_mapping_columns = (
    required_mapping_columns
    -
    set(
        mapping_df.columns
    )
)


if missing_mapping_columns:

    raise RuntimeError(
        "Missing mapping columns: "
        + str(
            sorted(
                missing_mapping_columns
            )
        )
    )


text_columns = [
    "feature_name",
    "forensic_source",
    "forensic_observation",
    "behavioral_hypothesis",
    "attack_candidate_id",
    "attack_candidate_name",
    "evidence_level",
    "note"
]


for col in text_columns:

    mapping_df[col] = (
        mapping_df[col]
        .fillna("")
        .astype(str)
    )


mapping_lookup = (
    mapping_df
    .set_index(
        "feature_name"
    )
    .to_dict(
        orient="index"
    )
)


missing_feature_rules = [

    feature

    for feature in feature_columns

    if feature not in mapping_lookup
]


if missing_feature_rules:

    raise RuntimeError(
        "Features missing from evidence mapping: "
        + str(
            missing_feature_rules
        )
    )


print(
    "[PASS] All 55 features have reviewed evidence rules."
)


# ============================================================
# 13. EXTRACT TOP-K PREDICTED-CLASS SHAP FEATURES
# ============================================================

feature_rows = []


for sample_pos in range(
    len(
        X_shap
    )
):

    true_id = int(
        y_shap[
            sample_pos
        ]
    )

    pred_id = int(
        pred_shap[
            sample_pos
        ]
    )


    true_family = (
        id_to_family[
            true_id
        ]
    )

    pred_family = (
        id_to_family[
            pred_id
        ]
    )


    confidence = float(
        prob_shap[
            sample_pos,
            pred_id
        ]
    )


    predicted_shap_values = (
        shap_values[
            sample_pos,
            :,
            pred_id
        ]
    )


    feature_values = X_shap[
        sample_pos
    ]


    order = np.argsort(
        np.abs(
            predicted_shap_values
        )
    )[::-1][
        :TOP_K
    ]


    positive_total = float(
        np.maximum(
            predicted_shap_values[
                order
            ],
            0.0
        ).sum()
    )


    for rank, feature_idx in enumerate(
        order,
        start=1
    ):

        feature_name = (
            feature_columns[
                feature_idx
            ]
        )


        shap_value = float(
            predicted_shap_values[
                feature_idx
            ]
        )


        abs_shap = float(
            abs(
                shap_value
            )
        )


        positive_shap = float(
            max(
                shap_value,
                0.0
            )
        )


        normalized_positive_support = (
            positive_shap
            /
            positive_total

            if positive_total > 0

            else 0.0
        )


        rule = (
            mapping_lookup[
                feature_name
            ]
        )


        attack_id = str(
            rule[
                "attack_candidate_id"
            ]
        ).strip()


        has_attack_candidate = (
            attack_id
            !=
            ""
        )


        supports_attack_candidate = (
            has_attack_candidate

            and

            shap_value > 0
        )


        feature_rows.append(
            {
                "Cohort_Position":
                    sample_pos,

                "Holdout_Position":
                    int(
                        selected_indices[
                            sample_pos
                        ]
                    ),

                "True_Family":
                    true_family,

                "Predicted_Family":
                    pred_family,

                "Prediction_Correct":
                    bool(
                        true_id
                        ==
                        pred_id
                    ),

                "Confidence":
                    confidence,

                "Rank":
                    rank,

                "Feature":
                    feature_name,

                "Feature_Value":
                    float(
                        feature_values[
                            feature_idx
                        ]
                    ),

                "SHAP_Value":
                    shap_value,

                "Absolute_SHAP":
                    abs_shap,

                "SHAP_Direction":
                    (
                        "SUPPORTS_EXPLAINED_CLASS"
                        if shap_value > 0

                        else
                        "OPPOSES_EXPLAINED_CLASS"
                        if shap_value < 0

                        else
                        "NEUTRAL"
                    ),

                "Positive_SHAP_Support":
                    positive_shap,

                "Normalized_Positive_SHAP_Support":
                    normalized_positive_support,

                "forensic_source":
                    rule[
                        "forensic_source"
                    ],

                "forensic_observation":
                    rule[
                        "forensic_observation"
                    ],

                "behavioral_hypothesis":
                    rule[
                        "behavioral_hypothesis"
                    ],

                "attack_candidate_id":
                    attack_id,

                "attack_candidate_name":
                    str(
                        rule[
                            "attack_candidate_name"
                        ]
                    ).strip(),

                "evidence_level":
                    str(
                        rule[
                            "evidence_level"
                        ]
                    ).strip(),

                "Has_ATTACK_Candidate":
                    has_attack_candidate,

                "Supports_ATTACK_Candidate":
                    supports_attack_candidate,

                "Direct_Detection_Claim":
                    False,

                "Causal_Claim":
                    False,

                "ATTACK_Ground_Truth":
                    False,

                "Analyst_Validation_Required":
                    True
            }
        )


feature_df = pd.DataFrame(
    feature_rows
)


# ============================================================
# 14. CASE-LEVEL ATT&CK SUPPORT
# ============================================================

case_summary_rows = []


for cohort_position, group in (
    feature_df
    .groupby(
        "Cohort_Position"
    )
):

    first = group.iloc[
        0
    ]


    positive_support_rows = group[
        group[
            "Supports_ATTACK_Candidate"
        ]
    ]


    total_positive_shap = float(
        group[
            "Positive_SHAP_Support"
        ].sum()
    )


    attack_positive_shap = float(
        positive_support_rows[
            "Positive_SHAP_Support"
        ].sum()
    )


    attack_normalized_support = (
        attack_positive_shap
        /
        total_positive_shap

        if total_positive_shap > 0

        else 0.0
    )


    candidate_ids = sorted(
        set(
            positive_support_rows[
                "attack_candidate_id"
            ]
            .astype(str)
        )
        -
        {""}
    )


    high_count = int(
        (
            positive_support_rows[
                "evidence_level"
            ]
            ==
            "HIGH_CANDIDATE"
        ).sum()
    )


    medium_count = int(
        (
            positive_support_rows[
                "evidence_level"
            ]
            ==
            "MEDIUM_CANDIDATE"
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


    case_summary_rows.append(
        {
            "Cohort_Position":
                int(
                    cohort_position
                ),

            "Holdout_Position":
                int(
                    first[
                        "Holdout_Position"
                    ]
                ),

            "True_Family":
                first[
                    "True_Family"
                ],

            "Predicted_Family":
                first[
                    "Predicted_Family"
                ],

            "Prediction_Correct":
                bool(
                    first[
                        "Prediction_Correct"
                    ]
                ),

            "Confidence":
                float(
                    first[
                        "Confidence"
                    ]
                ),

            "ATTACK_Candidate_Count":
                len(
                    candidate_ids
                ),

            "ATTACK_Candidate_IDs":
                "; ".join(
                    candidate_ids
                ),

            "Highest_Evidence_Level":
                highest_evidence,

            "High_Evidence_Feature_Count":
                high_count,

            "Medium_Evidence_Feature_Count":
                medium_count,

            "Total_Positive_SHAP":
                total_positive_shap,

            "ATTACK_Positive_SHAP":
                attack_positive_shap,

            "ATTACK_Normalized_SHAP_Support":
                attack_normalized_support,

            "Direct_Detection_Claim":
                False,

            "Causal_Claim":
                False,

            "ATTACK_Ground_Truth":
                False,

            "Analyst_Validation_Required":
                True
        }
    )


case_summary_df = pd.DataFrame(
    case_summary_rows
)


# ============================================================
# 15. CANDIDATE-SPECIFIC SUPPORT
# ============================================================

candidate_rows = []


candidate_feature_df = feature_df[
    feature_df[
        "Supports_ATTACK_Candidate"
    ]
].copy()


if len(
    candidate_feature_df
) > 0:

    grouped = (
        candidate_feature_df
        .groupby(
            [
                "Cohort_Position",
                "True_Family",
                "Predicted_Family",
                "attack_candidate_id",
                "attack_candidate_name"
            ]
        )
    )


    for keys, group in grouped:

        (
            cohort_position,
            true_family,
            predicted_family,
            attack_id,
            attack_name
        ) = keys


        positive_shap = float(
            group[
                "Positive_SHAP_Support"
            ].sum()
        )


        total_positive_shap = float(
            feature_df[
                feature_df[
                    "Cohort_Position"
                ]
                ==
                cohort_position
            ][
                "Positive_SHAP_Support"
            ].sum()
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


        candidate_rows.append(
            {
                "Cohort_Position":
                    int(
                        cohort_position
                    ),

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
                    int(
                        len(
                            group
                        )
                    ),

                "Supporting_Features":
                    "; ".join(
                        group[
                            "Feature"
                        ]
                        .astype(str)
                        .tolist()
                    ),

                "Candidate_Positive_SHAP":
                    positive_shap,

                "Candidate_Normalized_SHAP_Support":
                    normalized_support
            }
        )


candidate_support_df = pd.DataFrame(
    candidate_rows
)


# ============================================================
# 16. SUMMARY STATISTICS
# ============================================================

summary_rows = []


groups_to_summarize = {

    "ALL":
        case_summary_df,

    "BENIGN_TRUE":
        case_summary_df[
            case_summary_df[
                "True_Family"
            ]
            ==
            "Benign"
        ],

    "MALWARE_TRUE":
        case_summary_df[
            case_summary_df[
                "True_Family"
            ]
            !=
            "Benign"
        ],

    "CORRECT":
        case_summary_df[
            case_summary_df[
                "Prediction_Correct"
            ]
            ==
            True
        ],

    "ERROR":
        case_summary_df[
            case_summary_df[
                "Prediction_Correct"
            ]
            ==
            False
        ],
}


for group_name, subset in (
    groups_to_summarize.items()
):

    if len(
        subset
    ) == 0:

        continue


    values = (
        subset[
            "ATTACK_Normalized_SHAP_Support"
        ]
        .to_numpy(
            dtype=float
        )
    )


    summary_rows.append(
        {
            "Group":
                group_name,

            "N":
                len(
                    subset
                ),

            "Mean":
                float(
                    np.mean(
                        values
                    )
                ),

            "Median":
                float(
                    np.median(
                        values
                    )
                ),

            "Std":
                float(
                    np.std(
                        values,
                        ddof=1
                    )
                )
                if len(values) > 1
                else 0.0,

            "P25":
                float(
                    np.percentile(
                        values,
                        25
                    )
                ),

            "P75":
                float(
                    np.percentile(
                        values,
                        75
                    )
                ),

            "P90":
                float(
                    np.percentile(
                        values,
                        90
                    )
                ),

            "P95":
                float(
                    np.percentile(
                        values,
                        95
                    )
                ),

            "Max":
                float(
                    np.max(
                        values
                    )
                ),

            "Candidate_Case_Rate":
                float(
                    np.mean(
                        subset[
                            "ATTACK_Candidate_Count"
                        ]
                        >
                        0
                    )
                )
        }
    )


summary_stats_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# 17. PER-FAMILY SUPPORT SUMMARY
# ============================================================

family_summary_rows = []


for family_name, subset in (
    case_summary_df
    .groupby(
        "True_Family"
    )
):

    values = (
        subset[
            "ATTACK_Normalized_SHAP_Support"
        ]
        .to_numpy(
            dtype=float
        )
    )


    family_summary_rows.append(
        {
            "True_Family":
                family_name,

            "N":
                len(
                    subset
                ),

            "Mean_ATTACK_Support":
                float(
                    np.mean(
                        values
                    )
                ),

            "Median_ATTACK_Support":
                float(
                    np.median(
                        values
                    )
                ),

            "P75_ATTACK_Support":
                float(
                    np.percentile(
                        values,
                        75
                    )
                ),

            "P90_ATTACK_Support":
                float(
                    np.percentile(
                        values,
                        90
                    )
                ),

            "Candidate_Case_Rate":
                float(
                    np.mean(
                        subset[
                            "ATTACK_Candidate_Count"
                        ]
                        >
                        0
                    )
                ),

            "Prediction_Accuracy":
                float(
                    np.mean(
                        subset[
                            "Prediction_Correct"
                        ]
                    )
                )
        }
    )


family_summary_df = pd.DataFrame(
    family_summary_rows
)


# ============================================================
# 18. SCIENTIFIC SAFETY CHECKS
# ============================================================

print(
    "\n" + "=" * 96
)

print(
    "SCIENTIFIC SAFETY CHECKS"
)

print(
    "=" * 96
)


low_candidate_violation = feature_df[
    (
        feature_df[
            "evidence_level"
        ]
        ==
        "LOW_CONTEXTUAL"
    )
    &
    (
        feature_df[
            "Has_ATTACK_Candidate"
        ]
    )
]


assert (
    len(
        low_candidate_violation
    )
    ==
    0
)


print(
    "[PASS] LOW_CONTEXTUAL features cannot generate ATT&CK candidates."
)


negative_support_violation = feature_df[
    (
        feature_df[
            "SHAP_Value"
        ]
        <=
        0
    )
    &
    (
        feature_df[
            "Supports_ATTACK_Candidate"
        ]
    )
]


assert (
    len(
        negative_support_violation
    )
    ==
    0
)


print(
    "[PASS] Negative SHAP values cannot support ATT&CK candidates."
)


svcscan_violation = feature_df[
    (
        feature_df[
            "forensic_source"
        ]
        ==
        "svcscan"
    )
    &
    (
        feature_df[
            "Has_ATTACK_Candidate"
        ]
    )
]


assert (
    len(
        svcscan_violation
    )
    ==
    0
)


print(
    "[PASS] svcscan aggregate features remain contextual only."
)


assert (
    (
        case_summary_df[
            "ATTACK_Normalized_SHAP_Support"
        ]
        >=
        0
    ).all()
)


assert (
    (
        case_summary_df[
            "ATTACK_Normalized_SHAP_Support"
        ]
        <=
        1.0000001
    ).all()
)


print(
    "[PASS] Normalized ATT&CK support is bounded within [0, 1]."
)


# ============================================================
# 19. SAVE OUTPUTS
# ============================================================

feature_path = os.path.join(
    OUTPUT_DIR,
    "batch_local_shap_features.csv"
)

case_path = os.path.join(
    OUTPUT_DIR,
    "batch_attack_case_summary.csv"
)

candidate_path = os.path.join(
    OUTPUT_DIR,
    "batch_attack_candidate_support.csv"
)

summary_stats_path = os.path.join(
    OUTPUT_DIR,
    "attack_support_summary_statistics.csv"
)

family_summary_path = os.path.join(
    OUTPUT_DIR,
    "attack_support_by_family.csv"
)

cohort_distribution_path = os.path.join(
    OUTPUT_DIR,
    "shap_cohort_distribution.csv"
)

json_path = os.path.join(
    OUTPUT_DIR,
    "stage_viii_d1_summary.json"
)


feature_df.to_csv(
    feature_path,
    index=False
)


case_summary_df.to_csv(
    case_path,
    index=False
)


candidate_support_df.to_csv(
    candidate_path,
    index=False
)


summary_stats_df.to_csv(
    summary_stats_path,
    index=False
)


family_summary_df.to_csv(
    family_summary_path,
    index=False
)


cohort_distribution_df.to_csv(
    cohort_distribution_path,
    index=False
)


summary_json = {

    "stage":
        "VIII-D1",

    "name":
        (
            "Batch Local SHAP ATT&CK Support Analysis"
        ),

    "protocol": {

        "development_only":
            True,

        "group_aware":
            True,

        "splitter":
            "StratifiedGroupKFold",

        "n_splits":
            5,

        "random_state":
            RANDOM_STATE,

        "holdout_fold":
            "first_fold",

        "model":
            "Flat XGBoost",

        "shap_cohort_target":
            N_SHAP_SAMPLES,

        "shap_cohort_actual":
            int(
                len(
                    X_shap
                )
            ),

        "top_k":
            TOP_K
    },

    "model_performance": {

        "holdout_accuracy":
            float(
                holdout_acc
            ),

        "holdout_macro_f1":
            float(
                holdout_macro_f1
            )
    },

    "guardrails": {

        "attack_detection_claim":
            False,

        "causal_claim":
            False,

        "attack_ground_truth":
            False,

        "negative_shap_as_support":
            False,

        "low_contextual_generates_candidate":
            False,

        "analyst_validation_required":
            True
    }
}


with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary_json,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# 20. PRINT MAIN RESULTS
# ============================================================

print(
    "\n" + "=" * 96
)

print(
    "ATT&CK SUPPORT SUMMARY"
)

print(
    "=" * 96
)


print(
    summary_stats_df.to_string(
        index=False
    )
)


print(
    "\n" + "=" * 96
)

print(
    "PER-FAMILY SUMMARY"
)

print(
    "=" * 96
)


print(
    family_summary_df.to_string(
        index=False
    )
)


print(
    "\n" + "=" * 96
)

print(
    "FILES SAVED"
)

print(
    "=" * 96
)


print(
    "1.",
    feature_path
)

print(
    "2.",
    case_path
)

print(
    "3.",
    candidate_path
)

print(
    "4.",
    summary_stats_path
)

print(
    "5.",
    family_summary_path
)

print(
    "6.",
    cohort_distribution_path
)

print(
    "7.",
    json_path
)


print(
    "\nStage VIII-D1 complete."
)


print(
    "\nIMPORTANT:"
)

print(
    "ATT&CK support values quantify how much of the "
    "positive Top-K local SHAP support is associated "
    "with reviewed ATT&CK-aligned forensic features. "
    "They are not ATT&CK detection probabilities."
)