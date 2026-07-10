# VessShape Lab — Web App
# Convenience targets to run the FastAPI backend and the Vite/React frontend.
#
# Quick start:
#   make install     # backend deps (conda env) + frontend deps (npm)
#   make dev         # run backend + frontend together (Ctrl-C stops both)
#
# Run them separately in two terminals:
#   make backend
#   make frontend

# --- Config (override on the command line, e.g. `make backend BACKEND_PORT=9000`) ---
CONDA_ENV     ?= base
BACKEND_PORT  ?= 8000
FRONTEND_PORT ?= 5173

BACKEND_DIR   := backend
FRONTEND_DIR  := frontend

# `conda run -n <env>` keeps the target env active without needing a pre-activated shell.
CONDA_RUN     := conda run --no-capture-output -n $(CONDA_ENV)

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@echo "VessShape Lab — Web App"
	@echo
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Config: CONDA_ENV=$(CONDA_ENV) BACKEND_PORT=$(BACKEND_PORT) FRONTEND_PORT=$(FRONTEND_PORT)"

# --- Install ---------------------------------------------------------------
.PHONY: install install-backend install-frontend
install: install-backend install-frontend ## Install backend + frontend dependencies

install-backend: ## Install backend Python deps into the conda env
	$(CONDA_RUN) pip install -r $(BACKEND_DIR)/requirements.txt

install-frontend: ## Install frontend npm deps
	cd $(FRONTEND_DIR) && npm install

# --- Run -------------------------------------------------------------------
.PHONY: backend frontend dev
backend: ## Run the FastAPI backend (http://localhost:$(BACKEND_PORT))
	cd $(BACKEND_DIR) && $(CONDA_RUN) python -m uvicorn app.main:app --reload --port $(BACKEND_PORT)

frontend: ## Run the Vite dev server (http://localhost:$(FRONTEND_PORT))
	cd $(FRONTEND_DIR) && npm run dev -- --port $(FRONTEND_PORT)

dev: ## Run backend + frontend together (Ctrl-C stops both)
	@echo "Starting backend (:$(BACKEND_PORT)) and frontend (:$(FRONTEND_PORT))... Ctrl-C to stop both."
	@trap 'kill 0' EXIT INT TERM; \
	( cd $(BACKEND_DIR) && $(CONDA_RUN) python -m uvicorn app.main:app --reload --port $(BACKEND_PORT) ) & \
	( cd $(FRONTEND_DIR) && npm run dev -- --port $(FRONTEND_PORT) ) & \
	wait

# --- Build -----------------------------------------------------------------
.PHONY: build preview health
build: ## Production build of the frontend (outputs frontend/dist/)
	cd $(FRONTEND_DIR) && npm run build

preview: ## Preview the production build
	cd $(FRONTEND_DIR) && npm run preview -- --port $(FRONTEND_PORT)

health: ## Curl the backend health endpoint
	curl -fsS http://localhost:$(BACKEND_PORT)/api/health && echo

# --- Clean -----------------------------------------------------------------
.PHONY: clean clean-frontend
clean: clean-frontend ## Remove build artifacts and Python caches
	find $(BACKEND_DIR) -type d -name __pycache__ -prune -exec rm -rf {} +

clean-frontend: ## Remove the frontend production build
	rm -rf $(FRONTEND_DIR)/dist $(FRONTEND_DIR)/.vite
