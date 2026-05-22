SHELL := /bin/zsh

.PHONY: install smoke test fetch-fred fetch-treasury seed-source adapt-tdcest adapt-qrawatch build-quarterly-design tier2-component-credit tier2-component-outcomes tier2-credit-causality tier2-credit-lead-diagnostics tier2-missing40-attribution tier2-rolling-pass-through submission-appendix

install:
	set -a; source .env; set +a; uv pip install --python "$$UV_PROJECT_ENVIRONMENT/bin/python" -e '.[dev]'

smoke:
	set -a; source .env; set +a; "$$UV_PROJECT_ENVIRONMENT/bin/python" -B -m ea_tdc smoke

test:
	set -a; source .env; set +a; "$$UV_PROJECT_ENVIRONMENT/bin/python" -B -m pytest

fetch-fred:
	set -a; source .env; set +a; "$$UV_PROJECT_ENVIRONMENT/bin/python" -B -m ea_tdc fetch-fred

fetch-treasury:
	set -a; source .env; set +a; "$$UV_PROJECT_ENVIRONMENT/bin/python" -B -m ea_tdc fetch-treasury

seed-source:
	set -a; source .env; set +a; "$$UV_PROJECT_ENVIRONMENT/bin/python" -B -m ea_tdc seed-source

adapt-tdcest:
	set -a; source .env; set +a; "$$UV_PROJECT_ENVIRONMENT/bin/python" -B -m ea_tdc adapt-tdcest

adapt-qrawatch:
	set -a; source .env; set +a; "$$UV_PROJECT_ENVIRONMENT/bin/python" -B -m ea_tdc adapt-qrawatch

build-quarterly-design:
	set -a; source .env; set +a; "$$UV_PROJECT_ENVIRONMENT/bin/python" -B -m ea_tdc build-quarterly-design

tier2-component-credit:
	set -a; source .env; set +a; "$$UV_PROJECT_ENVIRONMENT/bin/python" -B scripts/run_tier2_canonical_component_credit_attribution.py

tier2-component-outcomes:
	set -a; source .env; set +a; "$$UV_PROJECT_ENVIRONMENT/bin/python" -B scripts/run_tier2_component_outcome_decomposition.py

tier2-credit-causality:
	set -a; source .env; set +a; "$$UV_PROJECT_ENVIRONMENT/bin/python" -B scripts/run_tier2_state_dependent_credit_causality.py

tier2-credit-lead-diagnostics:
	set -a; source .env; set +a; "$$UV_PROJECT_ENVIRONMENT/bin/python" -B scripts/run_tier2_credit_lead_diagnostics.py

tier2-missing40-attribution:
	set -a; source .env; set +a; "$$UV_PROJECT_ENVIRONMENT/bin/python" -B scripts/run_tier2_missing40_residual_attribution.py

tier2-rolling-pass-through:
	set -a; source .env; set +a; "$$UV_PROJECT_ENVIRONMENT/bin/python" -B scripts/run_tier2_rolling_pass_through.py

submission-appendix:
	set -a; source .env; set +a; "$$UV_PROJECT_ENVIRONMENT/bin/python" -B scripts/run_submission_appendix_diagnostics.py
