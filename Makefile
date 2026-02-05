.PHONY: help up down logs backend-shell register-model retrain daily-inference evaluate test lint format

help:
	@echo "Targets:"
	@echo "  up               Start full stack (API, UI, MLflow, MinIO, Prometheus, Grafana)"
	@echo "  down             Stop stack"
	@echo "  logs             Tail compose logs"
	@echo "  register-model   Log existing artifacts to MLflow + register model
	@echo "  retrain           True re-training pipeline (yfinance -> validate -> drift -> train -> register)""
	@echo "  daily-inference  Run batch inference locally (inside backend container recommended)"
	@echo "  evaluate         Run evaluation + push RMSE to Pushgateway"
	@echo "  test             Run unit tests"
	@echo "  format           Basic formatting (ruff/black not included by default)"

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f --tail=200

backend-shell:
	docker compose exec backend /bin/sh

register-model:
	docker compose exec backend python flows/train_and_register.py

daily-inference:
	docker compose exec backend python flows/daily_inference.py

evaluate:
	docker compose exec backend python flows/evaluate.py

test:
	docker compose exec backend pytest -q

format:
	@echo "Add ruff/black if you want auto-formatting."

retrain:
	docker compose exec backend python flows/retrain.py

lint:
	docker compose exec backend ruff check backend/src backend/flows backend/tests
