# SHM Evaluation Report

## Dataset

- Dataset: CWRU bearing family (Normal, Inner Race Fault, Outer Race Fault, Ball Fault)
- Window size: 1024 samples @ 12000 Hz
- Split: 70% train / 15% val / 15% test

## Results

| Model | Accuracy | F1 Weighted | ROC-AUC | False Alarm Rate |
|---|---:|---:|---:|---:|
| RandomForest | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| CNN1D | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
