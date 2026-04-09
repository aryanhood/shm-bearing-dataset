# Open-Source Benchmarks

This file records the repository patterns used to guide this project's open-source refactor.

## Reference repositories

1. scikit-learn
- Repo: https://github.com/scikit-learn/scikit-learn
- Contributor workflow: https://scikit-learn.org/stable/developers/contributing.html
- PR template: https://github.com/scikit-learn/scikit-learn/blob/main/.github/PULL_REQUEST_TEMPLATE.md
- Issue forms: https://github.com/scikit-learn/scikit-learn/tree/main/.github/ISSUE_TEMPLATE

2. JupyterHub
- Repo: https://github.com/jupyterhub/jupyterhub
- Dev setup docs: https://jupyterhub.readthedocs.io/en/latest/contributing/setup.html
- Project structure reference: https://github.com/jupyterhub/jupyterhub

3. Apache Airflow
- Repo: https://github.com/apache/airflow
- Contributor docs index: https://github.com/apache/airflow/blob/main/contributing-docs/README.rst
- Quick-start for contributors: https://github.com/apache/airflow/blob/main/contributing-docs/03a_contributors_quick_start_beginners.rst
- Issue/PR templates: https://github.com/apache/airflow/tree/main/.github

## Extracted patterns applied here

- Clear top-level separation between source code, docs, pipelines, tests, and artifacts.
- README focused on problem framing, user outcomes, and practical quickstart.
- Explicit contributor onboarding with local setup and quality expectations.
- Structured issue and PR templates to improve triage quality.
- Reproducibility discipline through run snapshots and configuration parity checks.
- Modularity through strict contracts between data, models, inference, and decision layers.
