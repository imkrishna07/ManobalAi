# Manobal-AI Synthetic Model Card

## Summary

- **Model:** calibrated LightGBM multiclass classifier
- **Model version:** `0.3.0-synthetic-backend-compatible`
- **Feature contract:** backend feature version `1.1.0`, 44 ordered features
- **Artifact training script:** `backend/train_synthetic_model.py`

## Intended use

Hackathon demonstration of the Manobal-AI inference and welfare-review workflow.
The model output is a simulated welfare signal for human review only. It is not
a clinical assessment, diagnosis, or real-personnel risk estimate.

## Synthetic data and labels

The training script reproducibly generates 1,500 synthetic personnel over 26
weeks (39,000 person-weeks). Labels are assigned from a hand-designed latent
strain simulation using generated workload, rest, leave, wellness, sleep,
biometric, and posting variables. Missingness is also simulated. There are no
real personnel records or validated real-world outcomes in this dataset.

Train, validation, and test sets are split by personnel: 975 / 225 / 300 people
respectively. The test set is held out by person. The calibration step uses only
the validation split.

## Model and preprocessing

LightGBM is fitted with balanced class weights, then sigmoid-calibrated on the
person-disjoint validation split. The baseline artifact is a class-balanced
logistic-regression pipeline. Raw weekly records are transformed through the
backend's canonical `lib.feature_engineering` pipeline; the saved preprocessing
artifact matches the backend inference interface.

## Evaluation on synthetic test labels

| Metric | Result |
|---|---:|
| Accuracy | 0.650 |
| Balanced accuracy | 0.581 |
| Macro F1 | 0.577 |
| Log loss | 0.758 |
| Low recall | 0.837 |
| Moderate recall | 0.331 |
| High recall | 0.574 |

These results measure recovery of the generated simulation labels only. They
must not be interpreted as real-world performance or as evidence of readiness
for operational use.

## Limitations and safeguards

- Synthetic labels are defined by the generator, not by validated welfare or
  clinical outcomes.
- No real-world validation, calibration study, or operational safety review
  has been performed.
- Moderate-band recall is limited even on this synthetic test set.
- The model should only support a human-led demo workflow and must not drive
  diagnosis, discipline, or automatic intervention.

Any use beyond a synthetic hackathon demonstration requires authorized data,
independent validation, subgroup analysis, privacy and security review, and
appropriate governance.
