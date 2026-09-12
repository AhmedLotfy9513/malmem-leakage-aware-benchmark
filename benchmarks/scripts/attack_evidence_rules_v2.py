# ============================================================
# Stage VIII-B v2
# Feature-Level Evidence-Aware MITRE ATT&CK Alignment
#
# Research principle:
#
# Memory Feature
#      ↓
# Forensic Observation
#      ↓
# Behavioral Hypothesis
#      ↓
# Optional ATT&CK Candidate
#      ↓
# Evidence Strength
#
# IMPORTANT:
# - ATT&CK candidates are NOT detections.
# - No causal claims are made.
# - Aggregate memory features are not ATT&CK ground truth.
# ============================================================

import os
import json
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = r"D:\lotfy\XPE_Memory_Project"

INPUT_INVENTORY = os.path.join(
    PROJECT_DIR,
    "outputs",
    "attack_alignment",
    "memory_feature_inventory.csv",
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
# 2. VALID EVIDENCE LEVELS
# ============================================================

VALID_EVIDENCE_LEVELS = {
    "HIGH_CANDIDATE",
    "MEDIUM_CANDIDATE",
    "LOW_CONTEXTUAL",
    "UNSUPPORTED",
}


# ============================================================
# 3. SOURCE-LEVEL CONTEXT
#
# This provides forensic meaning only.
# It does NOT automatically assign ATT&CK techniques.
# ============================================================

SOURCE_CONTEXT = {

    "malfind": {
        "observation":
            (
                "Memory-region characteristics associated "
                "with suspicious executable or injection-like "
                "memory activity."
            ),

        "general_hypothesis":
            (
                "Possible abnormal code placement or "
                "execution inside process memory."
            ),
    },

    "ldrmodules": {
        "observation":
            (
                "Discrepancies between expected Windows "
                "loader structures and observed modules."
            ),

        "general_hypothesis":
            (
                "Possible hidden, unlinked, manually mapped, "
                "or non-standard memory-resident modules."
            ),
    },

    "psxview": {
        "observation":
            (
                "Cross-view inconsistencies between multiple "
                "process-enumeration mechanisms."
            ),

        "general_hypothesis":
            (
                "Possible process concealment or "
                "rootkit-like hiding behavior."
            ),
    },

    "svcscan": {
        "observation":
            (
                "Aggregate Windows service and driver "
                "statistics recovered from memory."
            ),

        "general_hypothesis":
            (
                "Contextual information concerning Windows "
                "services and drivers."
            ),
    },

    "callbacks": {
        "observation":
            (
                "Kernel callback-related memory statistics."
            ),

        "general_hypothesis":
            (
                "Contextual kernel activity requiring "
                "additional forensic evidence."
            ),
    },

    "dlllist": {
        "observation":
            (
                "Aggregate DLL-loading statistics across "
                "processes."
            ),

        "general_hypothesis":
            (
                "General process module-loading context."
            ),
    },

    "modules": {
        "observation":
            (
                "Aggregate kernel-module statistics."
            ),

        "general_hypothesis":
            (
                "General kernel-module context."
            ),
    },

    "pslist": {
        "observation":
            (
                "Aggregate process, parent-process, thread, "
                "architecture, and handle statistics."
            ),

        "general_hypothesis":
            (
                "General process-execution context."
            ),
    },

    "handles": {
        "observation":
            (
                "Aggregate Windows object-handle statistics."
            ),

        "general_hypothesis":
            (
                "General operating-system resource "
                "interaction context."
            ),
    },
}


# ============================================================
# 4. FEATURE-LEVEL OVERRIDES
#
# THIS is the scientific correction compared with V1.
#
# A source no longer automatically assigns an ATT&CK ID.
# Only individually justified features receive a candidate.
# ============================================================

FEATURE_RULES = {

    # ========================================================
    # MALFIND
    # ========================================================

    "malfind.ninjections": {

        "behavioral_hypothesis":
            (
                "Observed injection-like memory regions may "
                "support a process-memory injection hypothesis."
            ),

        "attack_id":
            "T1055",

        "attack_name":
            "Process Injection",

        "evidence_level":
            "HIGH_CANDIDATE",

        "note":
            (
                "Strong supporting forensic evidence within "
                "this aggregate feature set, but not proof "
                "that T1055 occurred."
            ),
    },


    "malfind.uniqueInjections": {

        "behavioral_hypothesis":
            (
                "Distinct injection-like memory observations "
                "may support a process-memory injection "
                "hypothesis."
            ),

        "attack_id":
            "T1055",

        "attack_name":
            "Process Injection",

        "evidence_level":
            "HIGH_CANDIDATE",

        "note":
            (
                "One of the strongest injection-oriented "
                "features in the dataset; nevertheless, "
                "it remains supporting evidence rather "
                "than ATT&CK ground truth."
            ),
    },


    "malfind.commitCharge": {

        "behavioral_hypothesis":
            (
                "Memory-allocation characteristics associated "
                "with regions identified by memory analysis."
            ),

        "attack_id":
            "",

        "attack_name":
            "",

        "evidence_level":
            "LOW_CONTEXTUAL",

        "note":
            (
                "Allocation statistics cannot independently "
                "establish process injection."
            ),
    },


    "malfind.protection": {

        "behavioral_hypothesis":
            (
                "Memory-protection characteristics that may "
                "provide contextual information."
            ),

        "attack_id":
            "",

        "attack_name":
            "",

        "evidence_level":
            "LOW_CONTEXTUAL",

        "note":
            (
                "Protection-related aggregate statistics "
                "require additional memory evidence and "
                "cannot independently establish T1055."
            ),
    },


    # ========================================================
    # LDRMODULES
    #
    # Loader inconsistencies may be compatible with
    # non-standard in-memory loading.
    # ========================================================

    "ldrmodules.not_in_load": {
        "attack_id":
            "T1620",

        "attack_name":
            "Reflective Code Loading",

        "evidence_level":
            "MEDIUM_CANDIDATE",
    },

    "ldrmodules.not_in_init": {
        "attack_id":
            "T1620",

        "attack_name":
            "Reflective Code Loading",

        "evidence_level":
            "MEDIUM_CANDIDATE",
    },

    "ldrmodules.not_in_mem": {
        "attack_id":
            "T1620",

        "attack_name":
            "Reflective Code Loading",

        "evidence_level":
            "MEDIUM_CANDIDATE",
    },

    "ldrmodules.not_in_load_avg": {
        "attack_id":
            "T1620",

        "attack_name":
            "Reflective Code Loading",

        "evidence_level":
            "MEDIUM_CANDIDATE",
    },

    "ldrmodules.not_in_init_avg": {
        "attack_id":
            "T1620",

        "attack_name":
            "Reflective Code Loading",

        "evidence_level":
            "MEDIUM_CANDIDATE",
    },

    "ldrmodules.not_in_mem_avg": {
        "attack_id":
            "T1620",

        "attack_name":
            "Reflective Code Loading",

        "evidence_level":
            "MEDIUM_CANDIDATE",
    },
}


# ============================================================
# 5. ADD PSXVIEW FEATURE RULES
#
# All psxview features are direct cross-view discrepancy
# statistics. They can support a process-hiding/rootkit
# hypothesis, but cannot prove a rootkit.
# ============================================================

PSXVIEW_FEATURES = [

    "psxview.not_in_pslist",
    "psxview.not_in_eprocess_pool",
    "psxview.not_in_ethread_pool",
    "psxview.not_in_pspcid_list",
    "psxview.not_in_csrss_handles",
    "psxview.not_in_session",
    "psxview.not_in_deskthrd",

    "psxview.not_in_pslist_false_avg",
    "psxview.not_in_eprocess_pool_false_avg",
    "psxview.not_in_ethread_pool_false_avg",
    "psxview.not_in_pspcid_list_false_avg",
    "psxview.not_in_csrss_handles_false_avg",
    "psxview.not_in_session_false_avg",
    "psxview.not_in_deskthrd_false_avg",
]


for feature in PSXVIEW_FEATURES:

    FEATURE_RULES[feature] = {

        "behavioral_hypothesis":
            (
                "Cross-view process inconsistency may support "
                "a process-concealment or rootkit-like "
                "behavioral hypothesis."
            ),

        "attack_id":
            "T1014",

        "attack_name":
            "Rootkit",

        "evidence_level":
            "MEDIUM_CANDIDATE",

        "note":
            (
                "Cross-view inconsistencies are compatible "
                "with process hiding but have alternative "
                "forensic explanations. T1014 is therefore "
                "a candidate alignment only."
            ),
    }


# ============================================================
# 6. SVCSCAN CONSERVATIVE RULES
#
# IMPORTANT:
# These are aggregate counts.
#
# They do NOT establish that a Windows Service was created
# or modified.
#
# Therefore V2 intentionally provides NO direct
# T1543.003 candidate from these features alone.
# ============================================================

SVCSCAN_FEATURES = [

    "svcscan.nservices",
    "svcscan.kernel_drivers",
    "svcscan.fs_drivers",
    "svcscan.process_services",
    "svcscan.shared_process_services",
    "svcscan.interactive_process_services",
    "svcscan.nactive",
]


for feature in SVCSCAN_FEATURES:

    FEATURE_RULES[feature] = {

        "behavioral_hypothesis":
            (
                "Service or driver statistics provide "
                "context for potential service-related "
                "behavior."
            ),

        "attack_id":
            "",

        "attack_name":
            "",

        "evidence_level":
            "LOW_CONTEXTUAL",

        "note":
            (
                "Aggregate service or driver counts do not "
                "demonstrate creation or modification of a "
                "Windows Service. Additional event, registry, "
                "command, API, or object-level evidence would "
                "be required before considering T1543.003."
            ),
    }


# ============================================================
# 7. DEFAULT LOW-CONTEXTUAL SOURCES
# ============================================================

LOW_CONTEXTUAL_SOURCES = {

    "callbacks",
    "dlllist",
    "handles",
    "modules",
    "pslist",
}


# ============================================================
# 8. LOAD FEATURE INVENTORY
# ============================================================

print("=" * 80)
print("STAGE VIII-B V2 - FEATURE-LEVEL ATT&CK ALIGNMENT")
print("=" * 80)

inventory_df = pd.read_csv(
    INPUT_INVENTORY
)

print(
    "\nLoaded features:",
    len(inventory_df)
)


if len(inventory_df) != 55:

    raise RuntimeError(
        f"Expected 55 features, found {len(inventory_df)}."
    )


required_columns = {
    "feature_name",
    "forensic_source",
}


missing = (
    required_columns
    -
    set(inventory_df.columns)
)


if missing:

    raise RuntimeError(
        "Missing columns: "
        + str(sorted(missing))
    )


# ============================================================
# 9. BUILD FEATURE-LEVEL MAPPING
# ============================================================

rows = []


for _, row in inventory_df.iterrows():

    feature = row["feature_name"]

    source = row["forensic_source"]

    source_context = SOURCE_CONTEXT.get(
        source,
        {
            "observation":
                "Unreviewed forensic observation.",

            "general_hypothesis":
                "No reviewed behavioral hypothesis.",
        }
    )


    # --------------------------------------------------------
    # Explicit feature-level rule
    # --------------------------------------------------------

    if feature in FEATURE_RULES:

        rule = FEATURE_RULES[feature]

        behavioral_hypothesis = rule.get(
            "behavioral_hypothesis",
            source_context[
                "general_hypothesis"
            ],
        )

        attack_id = rule.get(
            "attack_id",
            "",
        )

        attack_name = rule.get(
            "attack_name",
            "",
        )

        evidence_level = rule.get(
            "evidence_level",
            "LOW_CONTEXTUAL",
        )

        note = rule.get(
            "note",
            (
                "Candidate mapping requires analyst "
                "validation and supporting evidence."
            ),
        )


    # --------------------------------------------------------
    # Default contextual evidence
    # --------------------------------------------------------

    elif source in LOW_CONTEXTUAL_SOURCES:

        behavioral_hypothesis = (
            source_context[
                "general_hypothesis"
            ]
        )

        attack_id = ""

        attack_name = ""

        evidence_level = (
            "LOW_CONTEXTUAL"
        )

        note = (
            "Aggregate contextual feature; no defensible "
            "feature-level ATT&CK candidate is assigned."
        )


    # --------------------------------------------------------
    # Fail-safe unsupported
    # --------------------------------------------------------

    else:

        behavioral_hypothesis = (
            "No validated feature-level behavioral "
            "hypothesis."
        )

        attack_id = ""

        attack_name = ""

        evidence_level = (
            "UNSUPPORTED"
        )

        note = (
            "Feature was not covered by the reviewed "
            "feature-level rule base."
        )


    rows.append(
        {
            "feature_name":
                feature,

            "forensic_source":
                source,

            "forensic_observation":
                source_context[
                    "observation"
                ],

            "behavioral_hypothesis":
                behavioral_hypothesis,

            "attack_candidate_id":
                attack_id,

            "attack_candidate_name":
                attack_name,

            "evidence_level":
                evidence_level,

            "direct_detection_allowed":
                False,

            "causal_claim_allowed":
                False,

            "analyst_validation_required":
                True,

            "note":
                note,
        }
    )


mapping_df = pd.DataFrame(
    rows
)


# ============================================================
# 10. VALIDATION
# ============================================================

invalid_levels = (
    set(
        mapping_df[
            "evidence_level"
        ]
    )
    -
    VALID_EVIDENCE_LEVELS
)


if invalid_levels:

    raise RuntimeError(
        "Invalid evidence levels: "
        + str(invalid_levels)
    )


# ============================================================
# 11. CRITICAL SAFETY ASSERTIONS
#
# These assertions prevent accidental scientific overclaiming.
# ============================================================

def get_row(feature_name):

    result = mapping_df[
        mapping_df[
            "feature_name"
        ]
        ==
        feature_name
    ]

    if len(result) != 1:

        raise RuntimeError(
            "Feature lookup failure: "
            + feature_name
        )

    return result.iloc[0]


# ------------------------------------------------------------
# MALFIND safety checks
# ------------------------------------------------------------

assert (
    get_row(
        "malfind.ninjections"
    )["evidence_level"]
    ==
    "HIGH_CANDIDATE"
)

assert (
    get_row(
        "malfind.uniqueInjections"
    )["evidence_level"]
    ==
    "HIGH_CANDIDATE"
)

assert (
    get_row(
        "malfind.commitCharge"
    )["attack_candidate_id"]
    ==
    ""
)

assert (
    get_row(
        "malfind.protection"
    )["attack_candidate_id"]
    ==
    ""
)


# ------------------------------------------------------------
# SVCSCAN safety check
#
# No svcscan feature may directly generate an ATT&CK
# candidate in this aggregate dataset.
# ------------------------------------------------------------

svcscan_candidate_count = int(
    (
        (
            mapping_df[
                "forensic_source"
            ]
            ==
            "svcscan"
        )
        &
        (
            mapping_df[
                "attack_candidate_id"
            ]
            .fillna("")
            .str.len()
            >
            0
        )
    ).sum()
)


assert (
    svcscan_candidate_count
    ==
    0
), (
    "Safety failure: svcscan aggregate features "
    "must not directly generate ATT&CK candidates."
)


# ------------------------------------------------------------
# Direct detection must be disabled everywhere
# ------------------------------------------------------------

assert (
    mapping_df[
        "direct_detection_allowed"
    ].sum()
    ==
    0
)


# ============================================================
# 12. SUMMARY
# ============================================================

evidence_summary = (
    mapping_df
    .groupby(
        [
            "forensic_source",
            "evidence_level",
        ]
    )
    .size()
    .reset_index(
        name="feature_count"
    )
)


candidate_summary = (
    mapping_df[
        mapping_df[
            "attack_candidate_id"
        ]
        .fillna("")
        .str.len()
        >
        0
    ]
    [
        [
            "feature_name",
            "forensic_source",
            "attack_candidate_id",
            "attack_candidate_name",
            "evidence_level",
        ]
    ]
    .copy()
)


# ============================================================
# 13. SAVE OUTPUTS
# ============================================================

mapping_path = os.path.join(
    OUTPUT_DIR,
    "attack_feature_evidence_mapping_v2.csv",
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "attack_source_evidence_summary_v2.csv",
)

candidate_path = os.path.join(
    OUTPUT_DIR,
    "attack_candidate_features_v2.csv",
)

json_path = os.path.join(
    OUTPUT_DIR,
    "attack_evidence_rules_v2.json",
)


mapping_df.to_csv(
    mapping_path,
    index=False,
)

evidence_summary.to_csv(
    summary_path,
    index=False,
)

candidate_summary.to_csv(
    candidate_path,
    index=False,
)


json_output = {

    "stage":
        "VIII-B-v2",

    "experiment":
        (
            "Feature-Level Evidence-Aware "
            "MITRE ATT&CK Alignment"
        ),

    "reasoning_chain": [
        "memory_feature",
        "forensic_observation",
        "behavioral_hypothesis",
        "optional_attack_candidate",
        "evidence_level",
        "analyst_validation",
    ],

    "guardrails": {

        "direct_attack_detection":
            False,

        "causal_claim":
            False,

        "attack_ground_truth":
            False,

        "source_level_auto_mapping":
            False,

        "feature_level_review":
            True,

        "analyst_validation_required":
            True,
    },

    "candidate_techniques": {

        "T1055":
            "Process Injection",

        "T1620":
            "Reflective Code Loading",

        "T1014":
            "Rootkit",
    },

    "intentionally_not_generated_from_aggregate_features": {

        "T1543.003":
            (
                "Windows Service candidate not generated "
                "from svcscan aggregate counts because "
                "creation/modification evidence is absent."
            )
    },
}


with open(
    json_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        json_output,
        file,
        indent=4,
        ensure_ascii=False,
    )


# ============================================================
# 14. PRINT RESULTS
# ============================================================

print("\n" + "=" * 80)
print("EVIDENCE DISTRIBUTION")
print("=" * 80)

print(
    evidence_summary.to_string(
        index=False
    )
)


print("\n" + "=" * 80)
print("FEATURES WITH ATT&CK CANDIDATES")
print("=" * 80)

if len(candidate_summary) == 0:

    print(
        "No candidate features."
    )

else:

    print(
        candidate_summary.to_string(
            index=False
        )
    )


print("\n" + "=" * 80)
print("SCIENTIFIC SAFETY CHECKS")
print("=" * 80)

print(
    "[PASS] Direct ATT&CK detection disabled."
)

print(
    "[PASS] malfind.commitCharge has no direct candidate."
)

print(
    "[PASS] malfind.protection has no direct candidate."
)

print(
    "[PASS] svcscan aggregate features generate no "
    "ATT&CK candidate."
)

print(
    "[PASS] Analyst validation required."
)


print("\nFiles saved:")

print(
    "1.",
    mapping_path
)

print(
    "2.",
    summary_path
)

print(
    "3.",
    candidate_path
)

print(
    "4.",
    json_path
)


print(
    "\nStage VIII-B v2 complete."
)