# Data Guide

## Modes

1. Synthetic mode (default)
- Controlled CWRU-like vibration generation
- No external download required
- Good for tests and reproducible demos

2. Real CWRU mode
- Place `.mat` files in `data/raw/`
- Set `data.use_synthetic: false` in `configs/config.yaml`

## Expected CWRU file naming

- `Normal_0.mat`, `Normal_1.mat`, `Normal_2.mat`
- `IR007_0.mat`, `IR014_0.mat`, `IR021_0.mat`
- `OR007@6_0.mat`, `OR014@6_0.mat`, `OR021@6_0.mat`
- `B007_0.mat`, `B014_0.mat`, `B021_0.mat`

## Data contract

- Input window size must match `data.window_size`
- Labels map to:
  - `0`: Normal
  - `1`: Inner Race Fault
  - `2`: Outer Race Fault
  - `3`: Ball Fault
