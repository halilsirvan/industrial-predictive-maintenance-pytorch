# Model Card: FailureMLP

## Intended use

Portfolio and educational demonstration of binary industrial equipment-failure
classification. The output may support exploration of maintenance thresholds;
it must not be used as a safety-critical instruction.

## Training data

UCI AI4I 2020 Predictive Maintenance Dataset (10,000 synthetic records, CC BY
4.0, DOI 10.24432/C5HS5C). Failure prevalence is 3.39%.

## Inputs

Product type, air temperature, process temperature, rotational speed, torque,
tool wear, temperature gap, estimated mechanical power, and a torque–wear
interaction. Identifiers, target, and failure-mode labels are excluded.

## Architecture and optimization

- Fully connected PyTorch network: input–32–16–1
- Batch normalization, ReLU, dropout
- AdamW optimizer
- Positive-class-weighted BCE-with-logits loss
- Validation PR-AUC monitoring
- Platt calibration on validation logits
- Validation-selected F1 threshold

## Evaluation

Evaluation uses a stratified 15% test partition untouched by preprocessing,
training, calibration, and threshold selection.

| Metric | Value |
|---|---:|
| ROC-AUC | 0.9737 |
| PR-AUC | 0.7670 |
| Precision | 0.6176 |
| Recall | 0.8235 |
| F1 | 0.7059 |
| Brier score | 0.0151 |

## Ethical and technical limitations

The data is synthetic, does not represent a specific workforce or protected
group, and is not suitable for personnel decisions. Performance may not
transfer to a real plant. Prediction concerns current-observation failure, not
future remaining useful life. Production use would require external validation,
time-aware evaluation, uncertainty analysis, drift monitoring, and integration
with engineering safety rules.

