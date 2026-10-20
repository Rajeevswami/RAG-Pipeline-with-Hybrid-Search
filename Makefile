.PHONY: install run test lint compose-up compose-down

install:
	python -m pip install -r requirements.txt

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest

lint:
	ruff check app scripts tests

compose-up:
	docker compose up --build

compose-down:
	docker compose down
