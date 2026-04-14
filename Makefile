SHELL := /bin/zsh

.PHONY: install smoke test fetch-fred fetch-treasury seed-source adapt-tdcest adapt-qrawatch build-quarterly-design

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
