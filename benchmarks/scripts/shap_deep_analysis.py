# ============================================================
# SHAP DEEP ANALYSIS
# CIC-MalMem-2022 - 16-Class Family Attribution
#
# PURPOSE
# ------------------------------------------------------------
# 1) Quantify per-family SHAP concentration:
#       - Top-3 importance share
#       - Top-5 importance share
#       - HHI concentration
#       - Normalized attribution entropy
#
# 2) Relate explanation concentration to family difficulty.
#
# 3) Reproduce the same group-aware SHAP split and explain
#    selected local cases:
#       - High-confidence correct prediction
#       - High-confidence misclassification
#
# IMPORTANT
# ------------------------------------------------------------
# SHAP explains MODEL BEHAVIOR.
# It does NOT establish malware-behavior causality.
# ============================================================

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from scipy.stats import spearmanr

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score
)

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

SHAP_DIR = os.path.join(
    PROJECT_DIR,
    "outputs",
    "shap_multiclass_analysis"
)

PER_FAMILY_SHAP_PATH = os.path.join(
    SHAP_DIR,
    "per_family_shap_importance.csv"
)

LOCAL_CASES_PATH = os.path.join(
    SHAP_DIR,
    "local_explanation_cases.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "outputs",
    "shap_deep_analysis"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


RANDOM_STATE = 42

N_SHAP_SAMPLES = 2000

EPS = 1e-12

TOP_LOCAL_FEATURES = 15


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 100)
print("SHAP DEEP ANALYSIS")
print("=" * 100)


df = pd.read_csv(
    DATA_PATH
)


dev = df[
    df["Split"] == "Development"
].copy()


print(
    f"Development rows: {len(dev)}"
)


# ============================================================
# 3. LABEL FUNCTION
# ============================================================

def get_family(category):

    category = str(category)

    if category == "Benign":
        return "Benign"

    parts = category.split("-")

    if len(parts) < 2:

        raise ValueError(
            f"Unexpected Category: {category}"
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
    f"Features: {len(feature_columns)}"
)

print(
    f"Families: {N_CLASSES}"
)


# ============================================================
# 6. LOAD PREVIOUS PER-FAMILY SHAP MATRIX
# ============================================================

if not os.path.exists(
    PER_FAMILY_SHAP_PATH
):

    raise FileNotFoundError(
        PER_FAMILY_SHAP_PATH
    )


per_family_shap_df = pd.read_csv(

    PER_FAMILY_SHAP_PATH,

    index_col=0
)


missing_families = [

    family

    for family in family_names

    if family not in per_family_shap_df.index
]


if missing_families:

    raise ValueError(

        "Families missing from SHAP matrix: "

        f"{missing_families}"
    )


missing_features = [

    feature

    for feature in feature_columns

    if feature not in per_family_shap_df.columns
]


if missing_features:

    raise ValueError(

        "Features missing from SHAP matrix: "

        f"{missing_features}"
    )


per_family_shap_df = (

    per_family_shap_df

    .loc[
        family_names,
        feature_columns
    ]
)


# ============================================================
# 7. SHAP CONCENTRATION METRICS
# ============================================================

concentration_rows = []


for family_name in family_names:

    importance = (

        per_family_shap_df
        .loc[
            family_name
        ]

        .to_numpy(
            dtype=float
        )
    )


    importance = np.maximum(

        importance,

        0.0
    )


    total = float(
        importance.sum()
    )


    if total <= EPS:

        raise ValueError(

            f"Near-zero SHAP total "
            f"for family {family_name}"
        )


    q = (

        importance

        /

        total
    )


    sorted_q = np.sort(
        q
    )[::-1]


    top1_share = float(
        sorted_q[
            :1
        ].sum()
    )


    top3_share = float(
        sorted_q[
            :3
        ].sum()
    )


    top5_share = float(
        sorted_q[
            :5
        ].sum()
    )


    top10_share = float(
        sorted_q[
            :10
        ].sum()
    )


    hhi = float(

        np.sum(
            q ** 2
        )
    )


    entropy = float(

        -np.sum(

            q

            *

            np.log(
                q + EPS
            )
        )
    )


    normalized_entropy = float(

        entropy

        /

        np.log(
            len(q)
        )
    )


    effective_features = float(

        1.0

        /

        hhi
    )


    concentration_rows.append(

        {

            "Family":
                family_name,

            "Top1_Share":
                top1_share,

            "Top3_Share":
                top3_share,

            "Top5_Share":
                top5_share,

            "Top10_Share":
                top10_share,

            "HHI":
                hhi,

            "Normalized_Entropy":
                normalized_entropy,

            "Effective_Feature_Count":
                effective_features
        }
    )


concentration_df = pd.DataFrame(
    concentration_rows
)


# ============================================================
# 8. REPRODUCE SAME GROUP-AWARE HOLDOUT
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


if (

    set(
        groups[
            train_idx
        ]
    )

    &

    set(
        groups[
            explain_idx
        ]
    )
):

    raise RuntimeError(
        "Group leakage detected."
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
    f"Train rows: {len(X_train)}"
)

print(
    f"Holdout rows: {len(X_holdout)}"
)


# ============================================================
# 9. RECREATE SAME FLAT XGBOOST
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


print("\nRetraining same Flat XGBoost...")

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


# ============================================================
# 10. PER-FAMILY PERFORMANCE ON SAME HOLDOUT
# ============================================================

performance_rows = []


for class_id, family_name in enumerate(
    family_names
):

    y_true_binary = (

        y_holdout
        ==
        class_id

    ).astype(int)


    y_pred_binary = (

        holdout_pred
        ==
        class_id

    ).astype(int)


    family_f1 = f1_score(

        y_true_binary,

        y_pred_binary,

        zero_division=0
    )


    family_precision = precision_score(

        y_true_binary,

        y_pred_binary,

        zero_division=0
    )


    family_recall = recall_score(

        y_true_binary,

        y_pred_binary,

        zero_division=0
    )


    support = int(

        np.sum(
            y_holdout
            ==
            class_id
        )
    )


    performance_rows.append(

        {

            "Family":
                family_name,

            "Support":
                support,

            "F1":
                float(
                    family_f1
                ),

            "Precision":
                float(
                    family_precision
                ),

            "Recall":
                float(
                    family_recall
                )
        }
    )


performance_df = pd.DataFrame(
    performance_rows
)


analysis_df = pd.merge(

    performance_df,

    concentration_df,

    on="Family",

    how="inner"
)


# ============================================================
# 11. CORRELATION ANALYSIS
# ============================================================

malware_only_df = analysis_df[

    analysis_df[
        "Family"
    ]

    !=

    "Benign"

].copy()


correlation_metrics = [

    "Top1_Share",
    "Top3_Share",
    "Top5_Share",
    "Top10_Share",
    "HHI",
    "Normalized_Entropy",
    "Effective_Feature_Count"
]


correlation_rows = []


for metric in correlation_metrics:

    rho, p_value = spearmanr(

        malware_only_df[
            metric
        ],

        malware_only_df[
            "F1"
        ]
    )


    correlation_rows.append(

        {

            "Metric":
                metric,

            "Spearman_Rho_with_F1":
                float(
                    rho
                ),

            "P_Value":
                float(
                    p_value
                )
        }
    )


correlation_df = pd.DataFrame(
    correlation_rows
)


# ============================================================
# 12. PRINT CONCENTRATION RESULTS
# ============================================================

print("\n")
print("=" * 100)
print("SHAP CONCENTRATION BY FAMILY")
print("=" * 100)


display_df = (

    analysis_df

    .sort_values(
        "HHI",
        ascending=False
    )
)


print(

    display_df[
        [
            "Family",
            "Support",
            "F1",
            "Top3_Share",
            "Top5_Share",
            "HHI",
            "Normalized_Entropy",
            "Effective_Feature_Count"
        ]
    ]

    .to_string(
        index=False
    )
)


print("\n")
print("=" * 100)
print("SHAP CONCENTRATION VS FAMILY F1")
print("=" * 100)


print(

    correlation_df

    .to_string(
        index=False
    )
)


# ============================================================
# 13. CONCENTRATION SCATTER PLOT
# ============================================================

plt.figure(
    figsize=(8, 6)
)


plt.scatter(

    malware_only_df[
        "HHI"
    ],

    malware_only_df[
        "F1"
    ]
)


for _, row in malware_only_df.iterrows():

    plt.annotate(

        row["Family"],

        (
            row["HHI"],
            row["F1"]
        ),

        fontsize=8,

        xytext=(
            3,
            3
        ),

        textcoords="offset points"
    )


plt.xlabel(
    "SHAP concentration (HHI)"
)

plt.ylabel(
    "Family one-vs-rest F1"
)

plt.title(
    "Family Performance vs SHAP Attribution Concentration"
)

plt.tight_layout()


scatter_path = os.path.join(

    OUTPUT_DIR,

    "family_f1_vs_shap_concentration.png"
)


plt.savefig(

    scatter_path,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# 14. RECREATE SAME STRATIFIED SHAP SUBSET
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


prob_shap = holdout_prob[
    selected_indices
]


pred_shap = holdout_pred[
    selected_indices
]


# ============================================================
# 15. LOAD PREVIOUSLY SELECTED LOCAL CASES
# ============================================================

local_cases_df = pd.read_csv(
    LOCAL_CASES_PATH
)


print("\n")
print("=" * 100)
print("LOCAL CASES TO EXPLAIN")
print("=" * 100)


print(
    local_cases_df.to_string(
        index=False
    )
)


# ============================================================
# 16. VALIDATE LOCAL CASE REPRODUCTION
# ============================================================

for _, case in local_cases_df.iterrows():

    sample_index = int(
        case[
            "Sample_Index"
        ]
    )


    reproduced_true = family_names[

        y_shap[
            sample_index
        ]
    ]


    reproduced_pred = family_names[

        pred_shap[
            sample_index
        ]
    ]


    if (

        reproduced_true

        !=

        case[
            "True_Family"
        ]

    ):

        raise RuntimeError(

            "True-family mismatch while "
            "reproducing local case."
        )


    if (

        reproduced_pred

        !=

        case[
            "Predicted_Family"
        ]

    ):

        raise RuntimeError(

            "Predicted-family mismatch while "
            "reproducing local case."
        )


print(
    "Local cases reproduced successfully."
)


# ============================================================
# 17. COMPUTE SHAP ONLY FOR SELECTED LOCAL CASES
# ============================================================

local_sample_indices = (

    local_cases_df[
        "Sample_Index"
    ]

    .astype(int)

    .tolist()
)


X_local = X_shap[
    local_sample_indices
]


explainer = shap.TreeExplainer(
    model
)


raw_local_shap = explainer.shap_values(
    X_local
)


if isinstance(
    raw_local_shap,
    list
):

    local_shap = np.stack(

        raw_local_shap,

        axis=2
    )

else:

    local_shap = np.asarray(
        raw_local_shap
    )


if (

    local_shap.ndim == 3

    and

    local_shap.shape[1]
    ==
    len(feature_columns)

    and

    local_shap.shape[2]
    ==
    N_CLASSES
):

    pass


elif (

    local_shap.ndim == 3

    and

    local_shap.shape[1]
    ==
    N_CLASSES

    and

    local_shap.shape[2]
    ==
    len(feature_columns)
):

    local_shap = np.transpose(

        local_shap,

        (
            0,
            2,
            1
        )
    )


else:

    raise ValueError(

        "Unexpected local SHAP shape: "

        f"{local_shap.shape}"
    )


print(
    f"Local SHAP shape: "
    f"{local_shap.shape}"
)


# ============================================================
# 18. EXTRACT TRUE VS PREDICTED CONTRIBUTIONS
# ============================================================

local_contribution_rows = []


for local_position, (
    _,
    case
) in enumerate(

    local_cases_df.iterrows()
):

    shap_sample_index = int(
        case[
            "Sample_Index"
        ]
    )


    true_id = int(

        y_shap[
            shap_sample_index
        ]
    )


    pred_id = int(

        pred_shap[
            shap_sample_index
        ]
    )


    true_family = family_names[
        true_id
    ]


    predicted_family = family_names[
        pred_id
    ]


    feature_values = X_shap[
        shap_sample_index
    ]


    true_shap_values = local_shap[

        local_position,

        :,

        true_id
    ]


    pred_shap_values = local_shap[

        local_position,

        :,

        pred_id
    ]


    # --------------------------------------------------------
    # Contributions to predicted class
    # --------------------------------------------------------

    predicted_order = np.argsort(

        np.abs(
            pred_shap_values
        )

    )[::-1][
        :TOP_LOCAL_FEATURES
    ]


    for rank, feature_idx in enumerate(

        predicted_order,

        start=1
    ):

        local_contribution_rows.append(

            {

                "Case":
                    case[
                        "Case"
                    ],

                "Sample_Index":
                    shap_sample_index,

                "True_Family":
                    true_family,

                "Predicted_Family":
                    predicted_family,

                "Explained_Class":
                    predicted_family,

                "Explanation_Type":
                    "Predicted_Class",

                "Rank":
                    rank,

                "Feature":
                    feature_columns[
                        feature_idx
                    ],

                "Feature_Value":
                    float(
                        feature_values[
                            feature_idx
                        ]
                    ),

                "SHAP_Value":
                    float(
                        pred_shap_values[
                            feature_idx
                        ]
                    ),

                "Absolute_SHAP":
                    float(
                        abs(
                            pred_shap_values[
                                feature_idx
                            ]
                        )
                    )
            }
        )


    # --------------------------------------------------------
    # Contributions to TRUE class
    # --------------------------------------------------------

    true_order = np.argsort(

        np.abs(
            true_shap_values
        )

    )[::-1][
        :TOP_LOCAL_FEATURES
    ]


    for rank, feature_idx in enumerate(

        true_order,

        start=1
    ):

        local_contribution_rows.append(

            {

                "Case":
                    case[
                        "Case"
                    ],

                "Sample_Index":
                    shap_sample_index,

                "True_Family":
                    true_family,

                "Predicted_Family":
                    predicted_family,

                "Explained_Class":
                    true_family,

                "Explanation_Type":
                    "True_Class",

                "Rank":
                    rank,

                "Feature":
                    feature_columns[
                        feature_idx
                    ],

                "Feature_Value":
                    float(
                        feature_values[
                            feature_idx
                        ]
                    ),

                "SHAP_Value":
                    float(
                        true_shap_values[
                            feature_idx
                        ]
                    ),

                "Absolute_SHAP":
                    float(
                        abs(
                            true_shap_values[
                                feature_idx
                            ]
                        )
                    )
            }
        )


local_contributions_df = pd.DataFrame(
    local_contribution_rows
)


# ============================================================
# 19. PRINT HIGH-CONFIDENCE ERROR EXPLANATION
# ============================================================

error_cases = local_cases_df[

    local_cases_df[
        "Case"
    ]

    .str.contains(
        "error",
        case=False,
        na=False
    )
]


if not error_cases.empty:

    error_case = error_cases.iloc[
        0
    ]


    error_sample_index = int(

        error_case[
            "Sample_Index"
        ]
    )


    error_rows = local_contributions_df[

        local_contributions_df[
            "Sample_Index"
        ]

        ==

        error_sample_index
    ]


    print("\n")
    print("=" * 100)
    print("HIGH-CONFIDENCE ERROR: PREDICTED-CLASS CONTRIBUTIONS")
    print("=" * 100)


    print(

        error_rows[

            error_rows[
                "Explanation_Type"
            ]

            ==

            "Predicted_Class"

        ][
            [
                "Rank",
                "Feature",
                "Feature_Value",
                "SHAP_Value"
            ]
        ]

        .head(
            10
        )

        .to_string(
            index=False
        )
    )


    print("\n")
    print("=" * 100)
    print("HIGH-CONFIDENCE ERROR: TRUE-CLASS CONTRIBUTIONS")
    print("=" * 100)


    print(

        error_rows[

            error_rows[
                "Explanation_Type"
            ]

            ==

            "True_Class"

        ][
            [
                "Rank",
                "Feature",
                "Feature_Value",
                "SHAP_Value"
            ]
        ]

        .head(
            10
        )

        .to_string(
            index=False
        )
    )


# ============================================================
# 20. LOCAL CONTRIBUTION PLOTS
# ============================================================

for _, case in local_cases_df.iterrows():

    sample_index = int(
        case[
            "Sample_Index"
        ]
    )


    case_rows = local_contributions_df[

        local_contributions_df[
            "Sample_Index"
        ]

        ==

        sample_index
    ]


    for explanation_type in [

        "Predicted_Class",

        "True_Class"
    ]:

        plot_df = case_rows[

            case_rows[
                "Explanation_Type"
            ]

            ==

            explanation_type

        ].head(
            10
        )


        plot_df = (

            plot_df

            .sort_values(
                "Absolute_SHAP",
                ascending=True
            )
        )


        plt.figure(
            figsize=(9, 6)
        )


        plt.barh(

            plot_df[
                "Feature"
            ],

            plot_df[
                "SHAP_Value"
            ]
        )


        explained_class = (

            plot_df[
                "Explained_Class"
            ]

            .iloc[
                0
            ]
        )


        plt.xlabel(
            "SHAP value"
        )


        plt.ylabel(
            "Memory feature"
        )


        plt.title(

            f"{case['Case']} - "
            f"Explanation for "
            f"{explained_class}"
        )


        plt.tight_layout()


        safe_case = (

            str(
                case[
                    "Case"
                ]
            )

            .replace(
                " ",
                "_"
            )

            .replace(
                "-",
                "_"
            )

            .lower()
        )


        safe_type = (

            explanation_type

            .lower()
        )


        plot_path = os.path.join(

            OUTPUT_DIR,

            f"{safe_case}_"
            f"{safe_type}.png"
        )


        plt.savefig(

            plot_path,

            dpi=300,

            bbox_inches="tight"
        )


        plt.close()


# ============================================================
# 21. SAVE RESULTS
# ============================================================

concentration_path = os.path.join(

    OUTPUT_DIR,

    "family_shap_concentration.csv"
)


correlation_path = os.path.join(

    OUTPUT_DIR,

    "shap_concentration_correlations.csv"
)


local_path = os.path.join(

    OUTPUT_DIR,

    "local_shap_contributions.csv"
)


analysis_df.to_csv(

    concentration_path,

    index=False
)


correlation_df.to_csv(

    correlation_path,

    index=False
)


local_contributions_df.to_csv(

    local_path,

    index=False
)


# ============================================================
# 22. SUMMARY JSON
# ============================================================

summary = {

    "families":
        int(
            len(
                analysis_df
            )
        ),

    "malware_families":
        int(
            len(
                malware_only_df
            )
        ),

    "highest_hhi_family":
        str(

            malware_only_df

            .sort_values(
                "HHI",
                ascending=False
            )

            .iloc[
                0
            ][
                "Family"
            ]
        ),

    "lowest_hhi_family":
        str(

            malware_only_df

            .sort_values(
                "HHI",
                ascending=True
            )

            .iloc[
                0
            ][
                "Family"
            ]
        )
}


summary_path = os.path.join(

    OUTPUT_DIR,

    "shap_deep_summary.json"
)


with open(

    summary_path,

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        summary,

        f,

        indent=4
    )


# ============================================================
# 23. FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 100)
print("FILES SAVED")
print("=" * 100)

print(
    concentration_path
)

print(
    correlation_path
)

print(
    scatter_path
)

print(
    local_path
)

print(
    summary_path
)


print("\n")
print("=" * 100)
print("SHAP DEEP ANALYSIS COMPLETE")
print("=" * 100)