# Contributing Guide

Thanks for contributing to the SHM Decision-Support System.

We welcome contributions across model quality, diagnostics, software architecture, testing, and documentation.

## Project goals

- Maintain reproducible SHM experiments
- Preserve clear data -> detection -> insight flow
- Keep API contracts stable and testable
- Improve real-world usefulness for maintenance decisions

## Development setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Run tests:

```bash
python -m pytest
```

## Contribution workflow

1. Open an issue (bug, feature, or docs).
2. Discuss scope before large changes.
3. Create a branch from `main`:
   - `feat/<short-name>`
   - `fix/<short-name>`
   - `docs/<short-name>`
4. Implement with tests and docs updates.
5. Open a pull request using the PR template.

## Quality expectations

- Add or update tests for behavioral changes.
- Keep modules single-responsibility and avoid hidden coupling.
- Maintain backward compatibility for public CLI/API behavior unless discussed.
- Keep generated artifacts out of source changes unless intentionally updating examples.

## Coding standards

- Python 3.11+
- Type hints for public functions and interfaces
- Clear names over clever implementations
- Keep side effects explicit in pipeline entrypoints

## Documentation expectations

For non-trivial changes, update relevant docs:

- `README.md` for user-facing workflow
- `docs/ARCHITECTURE.md` for system boundary changes
- `docs/ROADMAP.md` if project direction changes

## Pull request checklist

- [ ] Scope and motivation are described
- [ ] Tests pass locally
- [ ] New behavior has test coverage
- [ ] Docs are updated for user-visible changes
- [ ] No unrelated refactors included

## Good first contributions

- Better dataset validation and error messages
- New health-index calibration experiments
- Additional explainability views for `/explain`
- CI improvements and reproducibility checks
- Documentation quality improvements
