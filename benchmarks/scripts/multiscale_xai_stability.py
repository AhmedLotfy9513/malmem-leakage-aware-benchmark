# ============================================================
# Stage VII-B
# Multi-Scale Explanation Stability Analysis
#
# Evaluates:
#   1. SHAP Top-5 stability
#   2. SHAP Top-10 stability
#   3. SHAP full-ranking stability
#   4. Prediction stability
#   5. Correct vs Incorrect prediction stability
#
# Perturbation scales:
#   0.5%, 1%, 2%, 5% of training feature std
#
# IMPORTANT:
# - Development partition only
# - Group-aware outer split
# - Training-derived preprocessing/statistics only
# - No causal interpretation
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

rng = np.random.default_rng(SEED)


# ============================================================
# 2. PATHS
# ============================================================

PROJECT_DIR = r"D:\lotfy\XPE_Memory_Project"

DATA_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "MalMem_LockedSplit.csv",
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "outputs",
    "multiscale_xai_stability",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# 3. EXPERIMENT CONFIGURATION
# ============================================================

N_SAMPLES = 100

N_PERTURBATIONS = 20

PERTURBATION_SCALES = [
    0.005,
    0.010,
    0.020,
    0.050,
]

TOP_K_VALUES = [
    5,
    10,
]


# ============================================================
# 4. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 80)
print("STAGE VII-B - MULTI-SCALE XAI STABILITY")
print("=" * 80)

print("\nDataset shape:", df.shape)


# ============================================================
# 5. IDENTIFY REQUIRED COLUMNS
# ============================================================

split_candidates = [
    c
    for c in df.columns
    if c.lower()
    in {
        "split",
        "locked_split",
        "dataset_split",
        "partition",
        "split_label",
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
    "family_label",
]

group_candidates = [
    "Group_ID",
    "group_id",
    "GroupID",
    "Profile_ID",
    "profile_id",
]

FAMILY_COL = next(
    (
        c
        for c in family_candidates
        if c in df.columns
    ),
    None,
)

GROUP_COL = next(
    (
        c
        for c in group_candidates
        if c in df.columns
    ),
    None,
)

if FAMILY_COL is None:
    raise RuntimeError(
        "Family column not found."
    )

if GROUP_COL is None:
    raise RuntimeError(
        "Group_ID not found."
    )

print("Split column :", SPLIT_COL)
print("Family column:", FAMILY_COL)
print("Group column :", GROUP_COL)


# ============================================================
# 6. DEVELOPMENT PARTITION ONLY
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

if len(dev) == 0:
    raise RuntimeError(
        "Development subset is empty."
    )

print("\nDevelopment rows:", len(dev))


# ============================================================
# 7. IDENTIFY EXACTLY 55 FORENSIC FEATURES
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
    c
    for c in numeric_columns
    if c not in exclude_columns
]

metadata_names = {
    "row_id",
    "sample_id",
    "index",
    "fold",
    "seed",
}

feature_columns = [
    c
    for c in feature_columns
    if c.lower()
    not in metadata_names
]

print(
    "Detected feature count:",
    len(feature_columns),
)

if len(feature_columns) != 55:

    print("\nDetected numeric feature columns:")

    for feature in feature_columns:
        print(feature)

    raise RuntimeError(
        "Expected exactly 55 forensic features."
    )


# ============================================================
# 8. PREPARE LABELS AND GROUPS
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

class_names = (
    label_encoder.classes_
)

NUM_CLASSES = (
    len(class_names)
)

print("Number of classes:", NUM_CLASSES)

if NUM_CLASSES != 16:
    raise RuntimeError(
        f"Expected 16 classes, found {NUM_CLASSES}."
    )


# ============================================================
# 9. SAME GROUP-AWARE OUTER SPLIT
# ============================================================

outer_cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)

train_idx, test_idx = next(
    outer_cv.split(
        X,
        y,
        groups,
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

y_train = (
    y[
        train_idx
    ]
)

y_test = (
    y[
        test_idx
    ]
)

groups_train = (
    groups[
        train_idx
    ]
)

groups_test = (
    groups[
        test_idx
    ]
)

group_overlap = (
    set(groups_train)
    .intersection(
        set(groups_test)
    )
)

print("\nTrain rows:", len(train_idx))
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
# 11. TRAINING-ONLY PERTURBATION STATISTICS
# ============================================================

feature_min = (
    X_train
    .min(axis=0)
)

feature_max = (
    X_train
    .max(axis=0)
)

feature_std = (
    X_train
    .std(axis=0)
)

feature_std = (
    feature_std
    .replace(
        0,
        1.0,
    )
)

feature_range = (
    feature_max
    -
    feature_min
)

print(
    "\nZero-variance features:",
    int(
        (
            feature_range == 0
        ).sum()
    )
)


# ============================================================
# 12. TRAIN FLAT XGBOOST
# ============================================================

print("\nTraining Flat XGBoost...")

training_start = (
    time.perf_counter()
)

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
    tree_method="hist",
)

model.fit(
    X_train,
    y_train,
)

training_seconds = (
    time.perf_counter()
    -
    training_start
)

pred = model.predict(
    X_test
)

proba = model.predict_proba(
    X_test
)

accuracy = accuracy_score(
    y_test,
    pred,
)

macro_f1 = f1_score(
    y_test,
    pred,
    average="macro",
)

print(
    f"Accuracy : {accuracy:.6f}"
)

print(
    f"Macro-F1 : {macro_f1:.6f}"
)

print(
    f"Training : {training_seconds:.2f} sec"
)


# ============================================================
# 13. BUILD SHAP EXPLAINER
# ============================================================

print("\nBuilding SHAP TreeExplainer...")

explainer_start = (
    time.perf_counter()
)

explainer = shap.TreeExplainer(
    model
)

explainer_seconds = (
    time.perf_counter()
    -
    explainer_start
)

print(
    f"Explainer construction: "
    f"{explainer_seconds:.3f} sec"
)


# ============================================================
# 14. ROBUST SHAP EXTRACTION
# ============================================================

def get_shap_vector(
    sample_df,
    class_index,
):

    shap_values = (
        explainer.shap_values(
            sample_df
        )
    )

    # Older SHAP convention:
    # list of classes.
    if isinstance(
        shap_values,
        list,
    ):

        class_array = np.asarray(
            shap_values[
                class_index
            ]
        )

        return class_array[
            0
        ]

    shap_values = np.asarray(
        shap_values
    )

    # Newer multiclass convention:
    # samples x features x classes
    if shap_values.ndim == 3:

        if (
            shap_values.shape[2]
            ==
            NUM_CLASSES
        ):

            return shap_values[
                0,
                :,
                class_index
            ]

        # Fallback:
        # classes x samples x features
        if (
            shap_values.shape[0]
            ==
            NUM_CLASSES
        ):

            return shap_values[
                class_index,
                0,
                :
            ]

    # Fallback for unusual/binary shape.
    if shap_values.ndim == 2:

        return shap_values[
            0
        ]

    raise RuntimeError(
        "Unexpected SHAP shape: "
        f"{shap_values.shape}"
    )


# ============================================================
# 15. STABILITY FUNCTIONS
# ============================================================

def top_k_feature_indices(
    shap_vector,
    k,
):

    return np.argsort(
        np.abs(
            shap_vector
        )
    )[
        -k:
    ][::-1]


def calculate_jaccard(
    a,
    b,
):

    a = set(a)
    b = set(b)

    union = (
        a.union(b)
    )

    if len(union) == 0:
        return 1.0

    intersection = (
        a.intersection(b)
    )

    return (
        len(intersection)
        /
        len(union)
    )


def safe_spearman(
    a,
    b,
):

    rho, _ = spearmanr(
        a,
        b,
    )

    if np.isnan(rho):
        return 0.0

    return float(rho)


# ============================================================
# 16. SELECT A FIXED SAMPLE SET
#
# Same selected samples used at EVERY epsilon.
# This is important for paired comparison.
# ============================================================

all_test_indices = (
    np.arange(
        len(X_test)
    )
)

wrong_indices = (
    np.where(
        pred
        !=
        y_test
    )[0]
)

correct_indices = (
    np.where(
        pred
        ==
        y_test
    )[0]
)

print("\nCorrect test predictions:", len(correct_indices))
print("Incorrect test predictions:", len(wrong_indices))


# ------------------------------------------------------------
# We deliberately include both correct and incorrect samples.
# Approximately half/half where possible.
# ------------------------------------------------------------

target_wrong = min(
    50,
    len(wrong_indices),
)

target_correct = min(
    N_SAMPLES - target_wrong,
    len(correct_indices),
)

selected_wrong = (
    rng.choice(
        wrong_indices,
        size=target_wrong,
        replace=False,
    )
    if target_wrong > 0
    else np.array(
        [],
        dtype=int,
    )
)

selected_correct = (
    rng.choice(
        correct_indices,
        size=target_correct,
        replace=False,
    )
    if target_correct > 0
    else np.array(
        [],
        dtype=int,
    )
)

selected_indices = np.concatenate(
    [
        selected_wrong,
        selected_correct,
    ]
)

# Shuffle once.
rng.shuffle(
    selected_indices
)

print(
    "\nSelected stability samples:",
    len(selected_indices)
)

print(
    "Selected correct:",
    int(
        np.sum(
            pred[
                selected_indices
            ]
            ==
            y_test[
                selected_indices
            ]
        )
    )
)

print(
    "Selected incorrect:",
    int(
        np.sum(
            pred[
                selected_indices
            ]
            !=
            y_test[
                selected_indices
            ]
        )
    )
)


# ============================================================
# 17. CACHE ORIGINAL SHAP EXPLANATIONS
#
# Prevent unnecessary recalculation across epsilon levels.
# ============================================================

print(
    "\nComputing original SHAP explanations..."
)

original_cache = {}

for counter, local_idx in enumerate(
    selected_indices,
    start=1,
):

    sample = (
        X_test
        .iloc[
            [local_idx]
        ]
        .copy()
    )

    predicted_class = int(
        pred[
            local_idx
        ]
    )

    shap_vector = (
        get_shap_vector(
            sample,
            predicted_class,
        )
    )

    original_cache[
        int(local_idx)
    ] = {

        "predicted_class":
            predicted_class,

        "shap":
            shap_vector,

        "top5":
            top_k_feature_indices(
                shap_vector,
                5,
            ),

        "top10":
            top_k_feature_indices(
                shap_vector,
                10,
            ),
    }

    if counter % 20 == 0:

        print(
            f"Original SHAP: "
            f"{counter}/"
            f"{len(selected_indices)}"
        )


# ============================================================
# 18. MULTI-SCALE PERTURBATION EXPERIMENT
# ============================================================

detailed_rows = []

print(
    "\nRunning multi-scale perturbation analysis..."
)


for epsilon in PERTURBATION_SCALES:

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"Perturbation scale = "
        f"{epsilon:.3%}"
    )

    print(
        "=" * 80
    )

    for sample_counter, local_idx in enumerate(
        selected_indices,
        start=1,
    ):

        local_idx = int(
            local_idx
        )

        original = (
            X_test
            .iloc[
                [local_idx]
            ]
            .copy()
        )

        predicted_class = (
            original_cache[
                local_idx
            ][
                "predicted_class"
            ]
        )

        original_shap = (
            original_cache[
                local_idx
            ][
                "shap"
            ]
        )

        original_top5 = (
            original_cache[
                local_idx
            ][
                "top5"
            ]
        )

        original_top10 = (
            original_cache[
                local_idx
            ][
                "top10"
            ]
        )

        top5_scores = []
        top10_scores = []
        rank_scores = []
        prediction_scores = []
        confidence_changes = []

        original_confidence = float(
            proba[
                local_idx,
                predicted_class
            ]
        )

        for perturbation_id in range(
            N_PERTURBATIONS
        ):

            # ----------------------------------------------
            # Numerical local perturbation:
            #
            # x' = x + epsilon * sigma_train * N(0,1)
            #
            # This is a sensitivity test.
            # It is NOT claimed to be a physically valid
            # malware-memory intervention.
            # ----------------------------------------------

            noise = rng.normal(
                loc=0.0,
                scale=1.0,
                size=len(
                    feature_columns
                ),
            )

            perturbation = (
                noise
                *
                feature_std.values
                *
                epsilon
            )

            perturbed_values = (
                original
                .iloc[
                    0
                ]
                .values
                .astype(float)
                +
                perturbation
            )

            # Training-observed range constraint.
            perturbed_values = np.clip(
                perturbed_values,
                feature_min.values,
                feature_max.values,
            )

            perturbed = pd.DataFrame(
                [
                    perturbed_values
                ],
                columns=feature_columns,
            )

            perturbed_proba = (
                model.predict_proba(
                    perturbed
                )[0]
            )

            perturbed_prediction = int(
                np.argmax(
                    perturbed_proba
                )
            )

            prediction_scores.append(
                int(
                    perturbed_prediction
                    ==
                    predicted_class
                )
            )

            confidence_changes.append(
                float(
                    perturbed_proba[
                        predicted_class
                    ]
                    -
                    original_confidence
                )
            )

            perturbed_shap = (
                get_shap_vector(
                    perturbed,
                    predicted_class,
                )
            )

            perturbed_top5 = (
                top_k_feature_indices(
                    perturbed_shap,
                    5,
                )
            )

            perturbed_top10 = (
                top_k_feature_indices(
                    perturbed_shap,
                    10,
                )
            )

            top5_scores.append(
                calculate_jaccard(
                    original_top5,
                    perturbed_top5,
                )
            )

            top10_scores.append(
                calculate_jaccard(
                    original_top10,
                    perturbed_top10,
                )
            )

            rank_scores.append(
                safe_spearman(
                    np.abs(
                        original_shap
                    ),
                    np.abs(
                        perturbed_shap
                    ),
                )
            )

        is_correct = int(
            predicted_class
            ==
            int(
                y_test[
                    local_idx
                ]
            )
        )

        detailed_rows.append(
            {
                "Perturbation_Scale":
                    float(
                        epsilon
                    ),

                "Perturbation_Percent":
                    float(
                        epsilon
                        *
                        100
                    ),

                "Test_Local_Index":
                    local_idx,

                "True_Family":
                    class_names[
                        int(
                            y_test[
                                local_idx
                            ]
                        )
                    ],

                "Predicted_Family":
                    class_names[
                        predicted_class
                    ],

                "Correct":
                    is_correct,

                "Original_Confidence":
                    original_confidence,

                "Mean_Jaccard_Top5":
                    float(
                        np.mean(
                            top5_scores
                        )
                    ),

                "Std_Jaccard_Top5":
                    float(
                        np.std(
                            top5_scores
                        )
                    ),

                "Mean_Jaccard_Top10":
                    float(
                        np.mean(
                            top10_scores
                        )
                    ),

                "Std_Jaccard_Top10":
                    float(
                        np.std(
                            top10_scores
                        )
                    ),

                "Mean_SHAP_Rank_Spearman":
                    float(
                        np.mean(
                            rank_scores
                        )
                    ),

                "Std_SHAP_Rank_Spearman":
                    float(
                        np.std(
                            rank_scores
                        )
                    ),

                "Prediction_Stability":
                    float(
                        np.mean(
                            prediction_scores
                        )
                    ),

                "Mean_Confidence_Change":
                    float(
                        np.mean(
                            confidence_changes
                        )
                    ),

                "Mean_Absolute_Confidence_Change":
                    float(
                        np.mean(
                            np.abs(
                                confidence_changes
                            )
                        )
                    ),
            }
        )

        if sample_counter % 20 == 0:

            print(
                f"epsilon={epsilon:.3%} | "
                f"{sample_counter}/"
                f"{len(selected_indices)}"
            )


# ============================================================
# 19. SAVE SAMPLE-LEVEL RESULTS
# ============================================================

detailed_df = pd.DataFrame(
    detailed_rows
)

detailed_path = os.path.join(
    OUTPUT_DIR,
    "multiscale_stability_sample_level.csv",
)

detailed_df.to_csv(
    detailed_path,
    index=False,
)


# ============================================================
# 20. GLOBAL SUMMARY BY EPSILON
# ============================================================

global_summary_rows = []

for epsilon in PERTURBATION_SCALES:

    subset = detailed_df[
        np.isclose(
            detailed_df[
                "Perturbation_Scale"
            ],
            epsilon,
        )
    ]

    global_summary_rows.append(
        {
            "Perturbation_Scale":
                float(
                    epsilon
                ),

            "Perturbation_Percent":
                float(
                    epsilon
                    *
                    100
                ),

            "Samples":
                int(
                    len(subset)
                ),

            "Mean_Jaccard_Top5":
                float(
                    subset[
                        "Mean_Jaccard_Top5"
                    ].mean()
                ),

            "SD_Jaccard_Top5":
                float(
                    subset[
                        "Mean_Jaccard_Top5"
                    ].std()
                ),

            "Mean_Jaccard_Top10":
                float(
                    subset[
                        "Mean_Jaccard_Top10"
                    ].mean()
                ),

            "SD_Jaccard_Top10":
                float(
                    subset[
                        "Mean_Jaccard_Top10"
                    ].std()
                ),

            "Mean_SHAP_Rank_Spearman":
                float(
                    subset[
                        "Mean_SHAP_Rank_Spearman"
                    ].mean()
                ),

            "SD_SHAP_Rank_Spearman":
                float(
                    subset[
                        "Mean_SHAP_Rank_Spearman"
                    ].std()
                ),

            "Mean_Prediction_Stability":
                float(
                    subset[
                        "Prediction_Stability"
                    ].mean()
                ),

            "SD_Prediction_Stability":
                float(
                    subset[
                        "Prediction_Stability"
                    ].std()
                ),

            "Mean_Absolute_Confidence_Change":
                float(
                    subset[
                        "Mean_Absolute_Confidence_Change"
                    ].mean()
                ),
        }
    )


global_summary_df = pd.DataFrame(
    global_summary_rows
)

global_summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "multiscale_global_summary.csv",
    ),
    index=False,
)


# ============================================================
# 21. CORRECT VS INCORRECT SUMMARY
# ============================================================

correctness_rows = []

for epsilon in PERTURBATION_SCALES:

    epsilon_subset = detailed_df[
        np.isclose(
            detailed_df[
                "Perturbation_Scale"
            ],
            epsilon,
        )
    ]

    for correct_value, label in [
        (
            1,
            "Correct",
        ),
        (
            0,
            "Incorrect",
        ),
    ]:

        subset = epsilon_subset[
            epsilon_subset[
                "Correct"
            ]
            ==
            correct_value
        ]

        if len(subset) == 0:
            continue

        correctness_rows.append(
            {
                "Perturbation_Scale":
                    float(
                        epsilon
                    ),

                "Perturbation_Percent":
                    float(
                        epsilon
                        *
                        100
                    ),

                "Prediction_Group":
                    label,

                "Samples":
                    int(
                        len(subset)
                    ),

                "Mean_Jaccard_Top5":
                    float(
                        subset[
                            "Mean_Jaccard_Top5"
                        ].mean()
                    ),

                "Mean_Jaccard_Top10":
                    float(
                        subset[
                            "Mean_Jaccard_Top10"
                        ].mean()
                    ),

                "Mean_SHAP_Rank_Spearman":
                    float(
                        subset[
                            "Mean_SHAP_Rank_Spearman"
                        ].mean()
                    ),

                "Mean_Prediction_Stability":
                    float(
                        subset[
                            "Prediction_Stability"
                        ].mean()
                    ),

                "Mean_Absolute_Confidence_Change":
                    float(
                        subset[
                            "Mean_Absolute_Confidence_Change"
                        ].mean()
                    ),
            }
        )


correctness_df = pd.DataFrame(
    correctness_rows
)

correctness_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "correct_vs_incorrect_summary.csv",
    ),
    index=False,
)


# ============================================================
# 22. FAMILY-LEVEL SUMMARY
# ============================================================

family_rows = []

for epsilon in PERTURBATION_SCALES:

    epsilon_subset = detailed_df[
        np.isclose(
            detailed_df[
                "Perturbation_Scale"
            ],
            epsilon,
        )
    ]

    for family_name in class_names:

        subset = epsilon_subset[
            epsilon_subset[
                "True_Family"
            ]
            ==
            family_name
        ]

        if len(subset) == 0:
            continue

        family_rows.append(
            {
                "Perturbation_Scale":
                    float(
                        epsilon
                    ),

                "Perturbation_Percent":
                    float(
                        epsilon
                        *
                        100
                    ),

                "Family":
                    family_name,

                "Samples":
                    int(
                        len(subset)
                    ),

                "Mean_Jaccard_Top10":
                    float(
                        subset[
                            "Mean_Jaccard_Top10"
                        ].mean()
                    ),

                "Mean_SHAP_Rank_Spearman":
                    float(
                        subset[
                            "Mean_SHAP_Rank_Spearman"
                        ].mean()
                    ),

                "Mean_Prediction_Stability":
                    float(
                        subset[
                            "Prediction_Stability"
                        ].mean()
                    ),
            }
        )


family_df = pd.DataFrame(
    family_rows
)

family_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "family_level_stability.csv",
    ),
    index=False,
)


# ============================================================
# 23. PAPER-READY CURVE TABLE
# ============================================================

paper_curve_df = (
    global_summary_df[
        [
            "Perturbation_Percent",
            "Mean_Jaccard_Top5",
            "Mean_Jaccard_Top10",
            "Mean_SHAP_Rank_Spearman",
            "Mean_Prediction_Stability",
            "Mean_Absolute_Confidence_Change",
        ]
    ]
    .copy()
)

paper_curve_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "paper_stability_curve.csv",
    ),
    index=False,
)


# ============================================================
# 24. MASTER JSON SUMMARY
# ============================================================

summary = {

    "experiment":
        "Multi-Scale SHAP and Prediction Stability",

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
                training_seconds
            ),

        "shap_explainer_seconds":
            float(
                explainer_seconds
            ),
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
            ),
    },

    "stability_design": {

        "selected_samples":
            int(
                len(selected_indices)
            ),

        "selected_correct":
            int(
                np.sum(
                    pred[
                        selected_indices
                    ]
                    ==
                    y_test[
                        selected_indices
                    ]
                )
            ),

        "selected_incorrect":
            int(
                np.sum(
                    pred[
                        selected_indices
                    ]
                    !=
                    y_test[
                        selected_indices
                    ]
                )
            ),

        "perturbations_per_sample":
            int(
                N_PERTURBATIONS
            ),

        "perturbation_scales":
            [
                float(x)
                for x
                in PERTURBATION_SCALES
            ],

        "top_k_values":
            TOP_K_VALUES,
    },

    "global_results":
        global_summary_df
        .to_dict(
            orient="records"
        ),

    "interpretation_guardrails": {

        "perturbation_type":
            (
                "local numerical sensitivity "
                "using Gaussian noise scaled "
                "by training feature standard deviation"
            ),

        "training_range_clipping":
            True,

        "forensically_valid_intervention_claim":
            False,

        "shap_causal_claim":
            False,

        "purpose":
            (
                "measure local attribution "
                "and prediction sensitivity"
            ),
    },
}


with open(
    os.path.join(
        OUTPUT_DIR,
        "multiscale_xai_summary.json",
    ),
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=4,
    )


# ============================================================
# 25. PRINT RESULTS
# ============================================================

print("\n" + "=" * 80)
print("GLOBAL MULTI-SCALE RESULTS")
print("=" * 80)

print(
    global_summary_df[
        [
            "Perturbation_Percent",
            "Mean_Jaccard_Top5",
            "Mean_Jaccard_Top10",
            "Mean_SHAP_Rank_Spearman",
            "Mean_Prediction_Stability",
            "Mean_Absolute_Confidence_Change",
        ]
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 80)
print("CORRECT VS INCORRECT")
print("=" * 80)

print(
    correctness_df.to_string(
        index=False
    )
)


print("\nInterpretation note:")
print(
    "High SHAP stability does not automatically imply "
    "high prediction robustness."
)

print(
    "Perturbations are numerical local sensitivity tests, "
    "not causal or physically guaranteed malware-memory interventions."
)

print("\nResults saved to:")
print(OUTPUT_DIR)

print("\nStage VII-B complete.")