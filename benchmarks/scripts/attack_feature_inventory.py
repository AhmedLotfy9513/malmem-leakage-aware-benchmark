# ============================================================
# Stage VIII-A
# Memory-Forensics Feature Inventory for ATT&CK Alignment
#
# Purpose:
#   1. Load the audited CIC-MalMem-2022 working dataset.
#   2. Identify the exact 55 forensic features.
#   3. Group features by forensic source/plugin prefix.
#   4. Produce machine-readable inventories for later
#      evidence-aware MITRE ATT&CK alignment.
#
# IMPORTANT:
# - This script DOES NOT assign ATT&CK techniques.
# - No feature is treated as proof of malicious behavior.
# - No causal interpretation is performed.
# ============================================================

import os
import json
from collections import Counter

import pandas as pd


# ============================================================
# 1. PROJECT PATHS
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
    "attack_alignment",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 80)
print("STAGE VIII-A - MEMORY FEATURE INVENTORY")
print("=" * 80)

df = pd.read_csv(
    DATA_PATH
)

print("\nDataset:", DATA_PATH)
print("Rows   :", len(df))
print("Columns:", len(df.columns))


# ============================================================
# 3. METADATA COLUMNS TO EXCLUDE
# ============================================================

known_metadata = {
    "category",
    "class",
    "family",
    "family_label",
    "binary_label",
    "group_id",
    "groupid",
    "profile_id",
    "split",
    "locked_split",
    "dataset_split",
    "partition",
    "split_label",
    "row_id",
    "sample_id",
    "index",
    "fold",
    "seed",
}


# ============================================================
# 4. FIND NUMERIC FORENSIC FEATURES
# ============================================================

numeric_columns = (
    df
    .select_dtypes(
        include="number"
    )
    .columns
    .tolist()
)

feature_columns = [
    column
    for column in numeric_columns
    if column.lower()
    not in known_metadata
]


print(
    "\nDetected numeric forensic features:",
    len(feature_columns)
)


# ============================================================
# 5. SAFETY CHECK
# ============================================================

if len(feature_columns) != 55:

    print("\nWARNING:")
    print(
        "Expected 55 forensic features, "
        f"but detected {len(feature_columns)}."
    )

    print(
        "\nDetected candidate columns:"
    )

    for column in feature_columns:
        print(" -", column)

    raise RuntimeError(
        "Feature inventory mismatch. "
        "Stop before ATT&CK alignment."
    )


# ============================================================
# 6. FORENSIC PLUGIN / SOURCE IDENTIFICATION
#
# Example:
#   malfind.uniqueInjections -> malfind
#   pslist.nproc             -> pslist
#   handles.nfile            -> handles
#
# If no dot exists, group under 'unclassified'.
# ============================================================

def get_feature_source(
    feature_name
):

    if "." in feature_name:

        return (
            feature_name
            .split(
                ".",
                1,
            )[0]
            .strip()
            .lower()
        )

    return "unclassified"


feature_records = []

for feature in feature_columns:

    source = get_feature_source(
        feature
    )

    feature_records.append(
        {
            "feature_name":
                feature,

            "forensic_source":
                source,

            "attack_mapping_status":
                "UNREVIEWED",

            "behavioral_evidence_status":
                "UNREVIEWED",

            "notes":
                "",
        }
    )


inventory_df = pd.DataFrame(
    feature_records
)


# ============================================================
# 7. SOURCE COUNTS
# ============================================================

source_counts = Counter(
    inventory_df[
        "forensic_source"
    ]
)

source_summary = (
    pd.DataFrame(
        [
            {
                "forensic_source":
                    source,

                "feature_count":
                    count,
            }
            for source, count
            in sorted(
                source_counts.items()
            )
        ]
    )
)


# ============================================================
# 8. FEATURE STATISTICS
#
# These are descriptive only.
# They DO NOT imply forensic importance.
# ============================================================

stats_records = []

for feature in feature_columns:

    series = pd.to_numeric(
        df[feature],
        errors="coerce",
    )

    stats_records.append(
        {
            "feature_name":
                feature,

            "forensic_source":
                get_feature_source(
                    feature
                ),

            "missing_values":
                int(
                    series.isna().sum()
                ),

            "unique_values":
                int(
                    series.nunique(
                        dropna=True
                    )
                ),

            "minimum":
                (
                    float(
                        series.min()
                    )
                    if not series.dropna().empty
                    else None
                ),

            "maximum":
                (
                    float(
                        series.max()
                    )
                    if not series.dropna().empty
                    else None
                ),

            "mean":
                (
                    float(
                        series.mean()
                    )
                    if not series.dropna().empty
                    else None
                ),

            "std":
                (
                    float(
                        series.std()
                    )
                    if not series.dropna().empty
                    else None
                ),
        }
    )


stats_df = pd.DataFrame(
    stats_records
)


# ============================================================
# 9. BUILD SOURCE -> FEATURES DICTIONARY
# ============================================================

source_feature_map = {}

for source in sorted(
    inventory_df[
        "forensic_source"
    ].unique()
):

    source_feature_map[
        source
    ] = (
        inventory_df.loc[
            inventory_df[
                "forensic_source"
            ]
            ==
            source,
            "feature_name",
        ]
        .tolist()
    )


# ============================================================
# 10. SAVE CSV FILES
# ============================================================

inventory_path = os.path.join(
    OUTPUT_DIR,
    "memory_feature_inventory.csv",
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "forensic_source_summary.csv",
)

stats_path = os.path.join(
    OUTPUT_DIR,
    "memory_feature_statistics.csv",
)


inventory_df.to_csv(
    inventory_path,
    index=False,
)

source_summary.to_csv(
    summary_path,
    index=False,
)

stats_df.to_csv(
    stats_path,
    index=False,
)


# ============================================================
# 11. SAVE JSON MASTER INVENTORY
# ============================================================

master_json = {

    "stage":
        "VIII-A",

    "experiment":
        (
            "Memory-Forensics Feature "
            "Inventory for ATT&CK Alignment"
        ),

    "dataset":
        "CIC-MalMem-2022 working audited split",

    "feature_count":
        int(
            len(feature_columns)
        ),

    "forensic_source_count":
        int(
            len(source_feature_map)
        ),

    "forensic_sources":
        source_feature_map,

    "guardrails": {

        "attack_mapping_performed":
            False,

        "direct_feature_to_attack_claim":
            False,

        "causal_claim":
            False,

        "feature_importance_claim":
            False,

        "purpose":
            (
                "Inventory forensic observables "
                "before evidence-aware ATT&CK review."
            ),
    },
}


json_path = os.path.join(
    OUTPUT_DIR,
    "memory_feature_inventory.json",
)

with open(
    json_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        master_json,
        file,
        indent=4,
        ensure_ascii=False,
    )


# ============================================================
# 12. PRINT INVENTORY
# ============================================================

print("\n" + "=" * 80)
print("FORENSIC SOURCES")
print("=" * 80)

print(
    source_summary.to_string(
        index=False
    )
)


print("\n" + "=" * 80)
print("FEATURES BY SOURCE")
print("=" * 80)

for source, features in (
    source_feature_map.items()
):

    print(
        f"\n[{source}] "
        f"({len(features)} features)"
    )

    for feature in features:

        print(
            "  -",
            feature,
        )


print("\n" + "=" * 80)
print("GUARDRAIL")
print("=" * 80)

print(
    "No MITRE ATT&CK technique has been assigned."
)

print(
    "This stage only inventories forensic observables."
)

print(
    "ATT&CK alignment will be reviewed separately "
    "using evidence and behavioral hypotheses."
)


print("\nFiles saved:")

print(
    "1.",
    inventory_path,
)

print(
    "2.",
    summary_path,
)

print(
    "3.",
    stats_path,
)

print(
    "4.",
    json_path,
)

print("\nStage VIII-A complete.")