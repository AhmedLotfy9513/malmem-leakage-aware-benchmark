# ============================================================
# Stage VII
# Multi-Level Explainability for Malware Family Attribution
#
# 1. SHAP feature attribution
# 2. Explanation stability under small perturbations
# 3. Model-based counterfactual explanations
#
# IMPORTANT:
# - Development data only
# - Group-aware split
# - No causal claims
# - Counterfactuals are model explanations, not proof of
#   real malware behavior
# ============================================================

import os
import json
import random
import time
import warnings

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score

from xgboost import XGBClassifier

import shap

warnings.filterwarnings("ignore")


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# 2. PATHS
# ============================================================

PROJECT_DIR = r"D:\lotfy\XPE_Memory_Project"

DATA_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "MalMem_LockedSplit.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "outputs",
    "counterfactual_stability"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 3. CONFIGURATION
# ============================================================

N_STABILITY_SAMPLES = 100

N_PERTURBATIONS = 20

PERTURBATION_SCALE = 0.01

TOP_K = 10

N_COUNTERFACTUALS = 60

MAX_CF_FEATURES = 15


# ============================================================
# 4. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 80)
print("STAGE VII - COUNTERFACTUAL + EXPLANATION STABILITY")
print("=" * 80)

print("\nDataset shape:", df.shape)


# ============================================================
# 5. IDENTIFY REQUIRED COLUMNS
# ============================================================

split_candidates = [
    c for c in df.columns
    if c.lower() in {
        "split",
        "locked_split",
        "dataset_split",
        "partition",
        "split_label"
    }
]

if not split_candidates:
    raise RuntimeError(
        "Split column not found."
    )

SPLIT_COL = split_candidates[0]


family_candidates = [
    "Family",
    "family",
    "Family_Label",
    "family_label"
]

group_candidates = [
    "Group_ID",
    "group_id",
    "GroupID",
    "Profile_ID",
    "profile_id"
]

FAMILY_COL = next(
    (
        c for c in family_candidates
        if c in df.columns
    ),
    None
)

GROUP_COL = next(
    (
        c for c in group_candidates
        if c in df.columns
    ),
    None
)

if FAMILY_COL is None:
    raise RuntimeError(
        "Family column not found."
    )

if GROUP_COL is None:
    raise RuntimeError(
        "Group_ID not found."
    )


# ============================================================
# 6. DEVELOPMENT DATA ONLY
# ============================================================

split_text = (
    df[SPLIT_COL]
    .astype(str)
    .str.lower()
)

development_mask = (
    split_text.str.contains("dev")
    |
    split_text.str.contains("train")
)

dev = df.loc[
    development_mask
].copy()

print("Development rows:", len(dev))


# ============================================================
# 7. EXACTLY 55 FEATURES
# ============================================================

exclude_columns = {
    FAMILY_COL,
    GROUP_COL,
    SPLIT_COL,
    "Category",
    "category",
    "Class",
    "class",
    "Binary_Label",
    "binary_label",
}

numeric_columns = (
    dev.select_dtypes(
        include=[np.number]
    )
    .columns
    .tolist()
)

feature_columns = [
    c for c in numeric_columns
    if c not in exclude_columns
]

metadata_names = {
    "row_id",
    "sample_id",
    "index",
    "fold",
    "seed"
}

feature_columns = [
    c for c in feature_columns
    if c.lower() not in metadata_names
]

print("Feature count:", len(feature_columns))

if len(feature_columns) != 55:

    print("\nDetected:")
    for c in feature_columns:
        print(c)

    raise RuntimeError(
        "Expected exactly 55 forensic features."
    )


# ============================================================
# 8. LABEL ENCODING
# ============================================================

X = dev[
    feature_columns
].copy()

groups = (
    dev[GROUP_COL]
    .astype(str)
    .values
)

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    dev[FAMILY_COL]
    .astype(str)
    .values
)

class_names = label_encoder.classes_

NUM_CLASSES = len(class_names)

print("Classes:", NUM_CLASSES)

if NUM_CLASSES != 16:
    raise RuntimeError(
        "Expected 16 classes."
    )


# ============================================================
# 9. GROUP-AWARE HOLDOUT
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

train_idx, test_idx = next(
    cv.split(
        X,
        y,
        groups
    )
)

X_train_raw = (
    X.iloc[
        train_idx
    ]
    .copy()
)

X_test_raw = (
    X.iloc[
        test_idx
    ]
    .copy()
)

y_train = y[
    train_idx
]

y_test = y[
    test_idx
]

groups_train = groups[
    train_idx
]

groups_test = groups[
    test_idx
]

group_overlap = (
    set(groups_train)
    .intersection(
        set(groups_test)
    )
)

print("Train rows:", len(train_idx))
print("Test rows :", len(test_idx))
print("Group overlap:", len(group_overlap))

if len(group_overlap) != 0:
    raise RuntimeError(
        "Group leakage detected."
    )


# ============================================================
# 10. TRAINING-ONLY IMPUTATION
# ============================================================

train_medians = (
    X_train_raw
    .median(axis=0)
)

X_train = (
    X_train_raw
    .fillna(train_medians)
)

X_test = (
    X_test_raw
    .fillna(train_medians)
)


# ============================================================
# 11. FEATURE BOUNDS AND PERTURBATION SCALE
#
# Learned from TRAINING DATA ONLY.
# ============================================================

feature_min = (
    X_train.min(axis=0)
)

feature_max = (
    X_train.max(axis=0)
)

feature_std = (
    X_train.std(axis=0)
    .replace(0, 1.0)
)


# ============================================================
# 12. TRAIN SAME-SPLIT FLAT XGBOOST
# ============================================================

print("\nTraining Flat XGBoost...")

start = time.perf_counter()

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softprob",
    num_class=NUM_CLASSES,
    eval_metric="mlogloss",
    random_state=SEED,
    n_jobs=-1,
    tree_method="hist"
)

model.fit(
    X_train,
    y_train
)

training_time = (
    time.perf_counter()
    -
    start
)

pred = model.predict(
    X_test
)

proba = model.predict_proba(
    X_test
)

accuracy = accuracy_score(
    y_test,
    pred
)

macro_f1 = f1_score(
    y_test,
    pred,
    average="macro"
)

print(
    f"Accuracy : {accuracy:.6f}"
)

print(
    f"Macro-F1 : {macro_f1:.6f}"
)

print(
    f"Training : {training_time:.2f} sec"
)


# ============================================================
# 13. SHAP EXPLAINER
# ============================================================

print("\nBuilding SHAP explainer...")

explainer = shap.TreeExplainer(
    model
)


# ============================================================
# 14. ROBUST SHAP EXTRACTION
# ============================================================

def get_shap_vector(
    sample_df,
    class_index
):

    values = explainer.shap_values(
        sample_df
    )

    # Older SHAP:
    # list[class] -> samples x features
    if isinstance(
        values,
        list
    ):

        arr = np.asarray(
            values[class_index]
        )

        return arr[0]

    values = np.asarray(
        values
    )

    # Possible:
    # samples x features x classes
    if values.ndim == 3:

        if values.shape[2] == NUM_CLASSES:

            return values[
                0,
                :,
                class_index
            ]

        if values.shape[0] == NUM_CLASSES:

            return values[
                class_index,
                0,
                :
            ]

    # Binary / unusual fallback
    if values.ndim == 2:

        return values[0]

    raise RuntimeError(
        f"Unexpected SHAP shape: "
        f"{values.shape}"
    )


# ============================================================
# 15. TOP-K FEATURE FUNCTION
# ============================================================

def top_k_indices(
    shap_vector,
    k=TOP_K
):

    return np.argsort(
        np.abs(
            shap_vector
        )
    )[-k:][::-1]


# ============================================================
# 16. JACCARD FUNCTION
# ============================================================

def jaccard(
    a,
    b
):

    a = set(a)
    b = set(b)

    union = a.union(b)

    if len(union) == 0:
        return 1.0

    return (
        len(
            a.intersection(b)
        )
        /
        len(union)
    )


# ============================================================
# 17. STABILITY SAMPLE SELECTION
#
# Balanced-ish random sample from outer holdout.
# ============================================================

rng = np.random.default_rng(
    SEED
)

candidate_indices = np.arange(
    len(X_test)
)

if len(candidate_indices) > N_STABILITY_SAMPLES:

    stability_indices = (
        rng.choice(
            candidate_indices,
            size=N_STABILITY_SAMPLES,
            replace=False
        )
    )

else:

    stability_indices = (
        candidate_indices
    )


# ============================================================
# 18. EXPLANATION STABILITY
# ============================================================

print("\nRunning explanation stability analysis...")

stability_rows = []


for count, local_idx in enumerate(
    stability_indices,
    start=1
):

    original = (
        X_test.iloc[
            [local_idx]
        ]
        .copy()
    )

    predicted_class = int(
        pred[
            local_idx
        ]
    )

    original_shap = get_shap_vector(
        original,
        predicted_class
    )

    original_top = top_k_indices(
        original_shap,
        TOP_K
    )

    jaccard_scores = []

    rank_scores = []

    prediction_preserved = []

    for p in range(
        N_PERTURBATIONS
    ):

        perturbed = (
            original.copy()
        )

        noise = (
            rng.normal(
                loc=0.0,
                scale=1.0,
                size=len(
                    feature_columns
                )
            )
        )

        perturbation = (
            noise
            *
            feature_std.values
            *
            PERTURBATION_SCALE
        )

        perturbed.iloc[
            0,
            :
        ] = (
            perturbed.iloc[
                0,
                :
            ].values
            +
            perturbation
        )

        # Keep values inside ranges
        # observed in TRAINING data.
        perturbed.iloc[
            0,
            :
        ] = np.clip(
            perturbed.iloc[
                0,
                :
            ].values,
            feature_min.values,
            feature_max.values
        )

        perturbed_pred = int(
            model.predict(
                perturbed
            )[0]
        )

        prediction_preserved.append(
            int(
                perturbed_pred
                ==
                predicted_class
            )
        )

        perturbed_shap = (
            get_shap_vector(
                perturbed,
                predicted_class
            )
        )

        perturbed_top = (
            top_k_indices(
                perturbed_shap,
                TOP_K
            )
        )

        jaccard_scores.append(
            jaccard(
                original_top,
                perturbed_top
            )
        )

        rho, _ = spearmanr(
            np.abs(
                original_shap
            ),
            np.abs(
                perturbed_shap
            )
        )

        if np.isnan(rho):
            rho = 0.0

        rank_scores.append(
            rho
        )

    stability_rows.append(
        {
            "Test_Local_Index":
                int(local_idx),

            "True_Family":
                class_names[
                    y_test[
                        local_idx
                    ]
                ],

            "Predicted_Family":
                class_names[
                    predicted_class
                ],

            "Correct":
                int(
                    predicted_class
                    ==
                    y_test[
                        local_idx
                    ]
                ),

            "Mean_TopK_Jaccard":
                float(
                    np.mean(
                        jaccard_scores
                    )
                ),

            "Std_TopK_Jaccard":
                float(
                    np.std(
                        jaccard_scores
                    )
                ),

            "Mean_SHAP_Rank_Spearman":
                float(
                    np.mean(
                        rank_scores
                    )
                ),

            "Prediction_Stability":
                float(
                    np.mean(
                        prediction_preserved
                    )
                )
        }
    )

    if count % 10 == 0:

        print(
            f"Stability: "
            f"{count}/"
            f"{len(stability_indices)}"
        )


stability_df = pd.DataFrame(
    stability_rows
)

stability_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "explanation_stability.csv"
    ),
    index=False
)


# ============================================================
# 19. GLOBAL STABILITY SUMMARY
# ============================================================

stability_summary = {

    "samples":
        int(
            len(stability_df)
        ),

    "perturbations_per_sample":
        N_PERTURBATIONS,

    "perturbation_scale":
        PERTURBATION_SCALE,

    "top_k":
        TOP_K,

    "mean_topk_jaccard":
        float(
            stability_df[
                "Mean_TopK_Jaccard"
            ].mean()
        ),

    "std_topk_jaccard":
        float(
            stability_df[
                "Mean_TopK_Jaccard"
            ].std()
        ),

    "mean_rank_spearman":
        float(
            stability_df[
                "Mean_SHAP_Rank_Spearman"
            ].mean()
        ),

    "mean_prediction_stability":
        float(
            stability_df[
                "Prediction_Stability"
            ].mean()
        )
}


# ============================================================
# 20. COUNTERFACTUAL STRATEGY
#
# We do NOT invent arbitrary unconstrained values.
#
# For each selected test sample:
# 1. Identify a target family.
# 2. Find a real training example of that family.
# 3. Rank features by absolute difference.
# 4. Replace features progressively using REAL observed
#    target-sample values.
# 5. Stop when prediction changes to target family.
#
# This is a model-based feasible-value counterfactual.
# It is NOT causal proof.
# ============================================================

print("\nRunning counterfactual analysis...")


# ============================================================
# 21. SELECT COUNTERFACTUAL CASES
#
# Priority:
# - Misclassified samples
# - Then correctly classified samples
# ============================================================

wrong_indices = np.where(
    pred != y_test
)[0]

correct_indices = np.where(
    pred == y_test
)[0]

selected_cf_indices = []

n_wrong = min(
    len(wrong_indices),
    N_COUNTERFACTUALS // 2
)

if n_wrong > 0:

    selected_cf_indices.extend(
        rng.choice(
            wrong_indices,
            size=n_wrong,
            replace=False
        ).tolist()
    )

remaining = (
    N_COUNTERFACTUALS
    -
    len(
        selected_cf_indices
    )
)

if remaining > 0:

    remaining = min(
        remaining,
        len(correct_indices)
    )

    selected_cf_indices.extend(
        rng.choice(
            correct_indices,
            size=remaining,
            replace=False
        ).tolist()
    )


# ============================================================
# 22. TARGET PROTOTYPE SEARCH
# ============================================================

X_train_np = (
    X_train.values
    .astype(float)
)

# Scale only for DISTANCE computation.
# Counterfactual replacement values remain real raw values.

distance_scale = (
    feature_std.values
    .astype(float)
)

distance_scale[
    distance_scale == 0
] = 1.0


def nearest_target_sample(
    x,
    target_class
):

    target_mask = (
        y_train
        ==
        target_class
    )

    candidates = (
        X_train_np[
            target_mask
        ]
    )

    if len(candidates) == 0:

        return None

    distances = np.linalg.norm(
        (
            candidates
            -
            x
        )
        /
        distance_scale,
        axis=1
    )

    nearest_idx = np.argmin(
        distances
    )

    return candidates[
        nearest_idx
    ].copy()


# ============================================================
# 23. GENERATE GREEDY COUNTERFACTUAL
# ============================================================

def generate_counterfactual(
    x_series,
    original_prediction,
    target_class
):

    x = (
        x_series
        .values
        .astype(float)
        .copy()
    )

    target_sample = (
        nearest_target_sample(
            x,
            target_class
        )
    )

    if target_sample is None:

        return None

    standardized_diff = (
        np.abs(
            target_sample
            -
            x
        )
        /
        distance_scale
    )

    ranked_features = (
        np.argsort(
            standardized_diff
        )[::-1]
    )

    cf = x.copy()

    changed = []

    initial_proba = (
        model.predict_proba(
            pd.DataFrame(
                [x],
                columns=feature_columns
            )
        )[0]
    )

    initial_target_probability = float(
        initial_proba[
            target_class
        ]
    )

    for feature_idx in ranked_features[
        :MAX_CF_FEATURES
    ]:

        if np.isclose(
            cf[
                feature_idx
            ],
            target_sample[
                feature_idx
            ]
        ):

            continue

        old_value = float(
            cf[
                feature_idx
            ]
        )

        new_value = float(
            target_sample[
                feature_idx
            ]
        )

        cf[
            feature_idx
        ] = new_value

        changed.append(
            {
                "feature":
                    feature_columns[
                        feature_idx
                    ],

                "old_value":
                    old_value,

                "new_value":
                    new_value,

                "standardized_change":
                    float(
                        abs(
                            new_value
                            -
                            old_value
                        )
                        /
                        distance_scale[
                            feature_idx
                        ]
                    )
            }
        )

        cf_df = pd.DataFrame(
            [cf],
            columns=feature_columns
        )

        new_pred = int(
            model.predict(
                cf_df
            )[0]
        )

        if new_pred == target_class:

            final_proba = (
                model.predict_proba(
                    cf_df
                )[0]
            )

            final_target_probability = float(
                final_proba[
                    target_class
                ]
            )

            l1_standardized = float(
                np.sum(
                    np.abs(
                        cf
                        -
                        x
                    )
                    /
                    distance_scale
                )
            )

            l2_standardized = float(
                np.linalg.norm(
                    (
                        cf
                        -
                        x
                    )
                    /
                    distance_scale
                )
            )

            return {

                "success":
                    True,

                "features_changed":
                    len(changed),

                "changed_features":
                    changed,

                "standardized_L1":
                    l1_standardized,

                "standardized_L2":
                    l2_standardized,

                "initial_target_probability":
                    initial_target_probability,

                "final_target_probability":
                    final_target_probability
            }

    return {

        "success":
            False,

        "features_changed":
            len(changed),

        "changed_features":
            changed,

        "standardized_L1":
            float(
                np.sum(
                    np.abs(
                        cf
                        -
                        x
                    )
                    /
                    distance_scale
                )
            ),

        "standardized_L2":
            float(
                np.linalg.norm(
                    (
                        cf
                        -
                        x
                    )
                    /
                    distance_scale
                )
            ),

        "initial_target_probability":
            initial_target_probability,

        "final_target_probability":
            float(
                model.predict_proba(
                    pd.DataFrame(
                        [cf],
                        columns=feature_columns
                    )
                )[0][
                    target_class
                ]
            )
    }


# ============================================================
# 24. RUN COUNTERFACTUALS
#
# For misclassified samples:
# target = TRUE family.
#
# For correct samples:
# target = model's second most probable family.
# ============================================================

counterfactual_rows = []

counterfactual_detail = []


for count, local_idx in enumerate(
    selected_cf_indices,
    start=1
):

    x = X_test.iloc[
        local_idx
    ]

    true_class = int(
        y_test[
            local_idx
        ]
    )

    predicted_class = int(
        pred[
            local_idx
        ]
    )

    sample_proba = proba[
        local_idx
    ]

    if predicted_class != true_class:

        target_class = true_class

        target_reason = (
            "true_family_for_misclassification"
        )

    else:

        sorted_classes = np.argsort(
            sample_proba
        )[::-1]

        target_class = next(
            int(c)
            for c in sorted_classes
            if int(c)
            !=
            predicted_class
        )

        target_reason = (
            "second_highest_probability"
        )

    result = generate_counterfactual(
        x,
        predicted_class,
        target_class
    )

    if result is None:

        continue

    counterfactual_rows.append(
        {
            "Test_Local_Index":
                int(local_idx),

            "True_Family":
                class_names[
                    true_class
                ],

            "Original_Prediction":
                class_names[
                    predicted_class
                ],

            "Target_Family":
                class_names[
                    target_class
                ],

            "Target_Reason":
                target_reason,

            "Originally_Correct":
                int(
                    predicted_class
                    ==
                    true_class
                ),

            "Success":
                int(
                    result[
                        "success"
                    ]
                ),

            "Features_Changed":
                int(
                    result[
                        "features_changed"
                    ]
                ),

            "Standardized_L1":
                float(
                    result[
                        "standardized_L1"
                    ]
                ),

            "Standardized_L2":
                float(
                    result[
                        "standardized_L2"
                    ]
                ),

            "Initial_Target_Probability":
                float(
                    result[
                        "initial_target_probability"
                    ]
                ),

            "Final_Target_Probability":
                float(
                    result[
                        "final_target_probability"
                    ]
                )
        }
    )

    counterfactual_detail.append(
        {
            "test_local_index":
                int(local_idx),

            "true_family":
                class_names[
                    true_class
                ],

            "original_prediction":
                class_names[
                    predicted_class
                ],

            "target_family":
                class_names[
                    target_class
                ],

            "success":
                bool(
                    result[
                        "success"
                    ]
                ),

            "changed_features":
                result[
                    "changed_features"
                ]
        }
    )

    if count % 10 == 0:

        print(
            f"Counterfactuals: "
            f"{count}/"
            f"{len(selected_cf_indices)}"
        )


counterfactual_df = pd.DataFrame(
    counterfactual_rows
)

counterfactual_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "counterfactual_summary.csv"
    ),
    index=False
)

with open(
    os.path.join(
        OUTPUT_DIR,
        "counterfactual_details.json"
    ),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        counterfactual_detail,
        f,
        indent=4
    )


# ============================================================
# 25. COUNTERFACTUAL GLOBAL SUMMARY
# ============================================================

if len(counterfactual_df) > 0:

    successful = (
        counterfactual_df[
            counterfactual_df[
                "Success"
            ] == 1
        ]
    )

    success_rate = float(
        counterfactual_df[
            "Success"
        ].mean()
    )

    if len(successful) > 0:

        median_features_changed = float(
            successful[
                "Features_Changed"
            ].median()
        )

        mean_features_changed = float(
            successful[
                "Features_Changed"
            ].mean()
        )

        median_l1 = float(
            successful[
                "Standardized_L1"
            ].median()
        )

        median_l2 = float(
            successful[
                "Standardized_L2"
            ].median()
        )

    else:

        median_features_changed = None
        mean_features_changed = None
        median_l1 = None
        median_l2 = None

else:

    success_rate = 0.0
    median_features_changed = None
    mean_features_changed = None
    median_l1 = None
    median_l2 = None


counterfactual_summary = {

    "attempted":
        int(
            len(
                counterfactual_df
            )
        ),

    "success_rate":
        success_rate,

    "median_features_changed_success":
        median_features_changed,

    "mean_features_changed_success":
        mean_features_changed,

    "median_standardized_L1_success":
        median_l1,

    "median_standardized_L2_success":
        median_l2
}


# ============================================================
# 26. MASTER SUMMARY
# ============================================================

master_summary = {

    "experiment":
        "Multi-Level XAI: SHAP Stability and Model Counterfactuals",

    "seed":
        SEED,

    "model": {

        "type":
            "Flat XGBoost",

        "accuracy":
            float(
                accuracy
            ),

        "macro_f1":
            float(
                macro_f1
            ),

        "training_seconds":
            float(
                training_time
            )
    },

    "data_integrity": {

        "development_rows":
            int(
                len(dev)
            ),

        "train_rows":
            int(
                len(train_idx)
            ),

        "test_rows":
            int(
                len(test_idx)
            ),

        "group_overlap":
            int(
                len(group_overlap)
            ),

        "features":
            int(
                len(feature_columns)
            ),

        "classes":
            int(
                NUM_CLASSES
            )
    },

    "stability":
        stability_summary,

    "counterfactual":
        counterfactual_summary,

    "interpretation_guardrail": {

        "shap_is_causal":
            False,

        "counterfactual_is_causal":
            False,

        "counterfactual_type":
            "model-based feasible-value counterfactual",

        "attack_mapping_status":
            "not_performed_in_this_stage"
    }
}


with open(
    os.path.join(
        OUTPUT_DIR,
        "xai_summary.json"
    ),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        master_summary,
        f,
        indent=4
    )


# ============================================================
# 27. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 80)
print("EXPLANATION STABILITY RESULTS")
print("=" * 80)

print(
    "Mean Top-K Jaccard:",
    f"{stability_summary['mean_topk_jaccard']:.6f}"
)

print(
    "Mean SHAP Rank Spearman:",
    f"{stability_summary['mean_rank_spearman']:.6f}"
)

print(
    "Prediction Stability:",
    f"{stability_summary['mean_prediction_stability']:.6f}"
)


print("\n" + "=" * 80)
print("COUNTERFACTUAL RESULTS")
print("=" * 80)

print(
    "Attempted:",
    counterfactual_summary[
        "attempted"
    ]
)

print(
    "Success rate:",
    f"{counterfactual_summary['success_rate']:.6f}"
)

print(
    "Median features changed:",
    counterfactual_summary[
        "median_features_changed_success"
    ]
)

print(
    "Median standardized L1:",
    counterfactual_summary[
        "median_standardized_L1_success"
    ]
)


print("\nIMPORTANT INTERPRETATION:")
print(
    "These are MODEL-BASED explanations."
)

print(
    "They are NOT causal malware-behavior claims."
)

print(
    "MITRE ATT&CK mapping will be performed "
    "separately using an evidence-aware layer."
)


print("\nSaved to:")
print(OUTPUT_DIR)

print("\nStage VII complete.")