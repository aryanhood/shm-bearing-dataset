# Roadmap

## Vision

Evolve this repository into a reliable SHM platform that supports both:

- research-grade experiments
- deployment-ready engineering diagnostics

## Milestone 1: Core system hardening (current)

- [x] Unified inference pipeline across CLI/API
- [x] Structured contracts for prediction and health status
- [x] Artifact manifest compatibility checks
- [x] Reproducible train/evaluate/predict/ablation workflows
- [x] Open-source onboarding docs and templates

## Milestone 2: SHM model quality and explainability

- [ ] Add temporal-context models (sequence windows)
- [ ] Add per-sample explainability (e.g., SHAP for RF)
- [ ] Calibrate health index against fault severity curves
- [ ] Add uncertainty-aware prediction outputs

## Milestone 3: Dataset and benchmark expansion

- [ ] Add dataset adapters beyond CWRU
- [ ] Provide benchmark scripts for cross-dataset generalization
- [ ] Track reproducibility matrices (seed x model x feature set)

## Milestone 4: Production-readiness

- [ ] Containerized deployment profile
- [ ] CI pipeline with lint/test/docs gates
- [ ] Model registry metadata and artifact retention policy
- [ ] Runtime monitoring hooks and inference telemetry

## Milestone 5: Industry integration

- [ ] Integrate with CMMS/maintenance workflows
- [ ] Alert routing by urgency class
- [ ] Fleet-level trend dashboards and degradation timelines

## Contribution opportunities

High-impact contribution areas:

1. Robust fault simulation and data quality validation
2. Domain-informed feature engineering
3. Decision policy tuning and false-alarm reduction
4. Scalable deployment and observability
