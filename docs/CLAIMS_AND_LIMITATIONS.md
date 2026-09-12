# Claims and Limitations

## Supported Claims

The package supports the following claims:

1. Leakage-aware grouping prevents known feature-identical profiles
   from crossing the locked Development / Locked_Test boundary.

2. The locked split contains no Group_ID overlap between Development
   and Locked_Test.

3. Binary malware detection on CIC-MalMem-2022 is highly saturated
   under the evaluated models.

4. Malware-family attribution remains substantially more difficult
   than binary malware detection.

5. Nested static fusion improves family-balanced Macro-F1 relative to
   the evaluated flat and hierarchical baselines under the rigorous
   protocol.

6. Weighted or micro-averaged metrics can obscure poor family-balanced
   performance in multiclass malware attribution.

7. SHAP explanation concentration is associated with family-level
   performance in exploratory analysis.

8. Misclassified samples exhibit lower local explanation stability
   and prediction stability than correctly classified samples in the
   evaluated balanced diagnostic cohort.

9. Evidence-aware ATT&CK alignment can characterize model-supported
   behavioral hypotheses without claiming ground-truth ATT&CK
   technique detection.

## Claims That Are NOT Supported

This package does NOT establish:

- state-of-the-art sixteen-class performance
- zero-day malware detection
- causal interpretation of SHAP values
- confirmed ATT&CK technique detection
- physical feasibility of all model counterfactuals
- superiority of reduced seven-feature models over the full model
  in overall predictive performance
- end-to-end operational latency from memory acquisition to analyst
  report using model inference timing alone
- that contrastive learning improved the primary benchmark
- that Gaussian perturbations represent physically realizable malware
  interventions

## Evaluation Boundaries

Conventional random-split results are included only for comparison
with prior literature.

The primary scientific conclusions should rely on leakage-aware,
group-aware evaluation.

Oracle experiments are diagnostic upper-bound analyses and must not
be interpreted as deployable models.

ATT&CK candidate support is an evidence-characterization measure,
not a detection probability.