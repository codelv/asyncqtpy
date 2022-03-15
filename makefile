isort:
	isort --profile=black ./

typecheck:
	mypy asyncqtpy --ignore-missing-imports
lintcheck:
	flake8 --ignore=E501,W503 ./
reformat:
	black ./
test:
	pytest -v tests --cov asyncqtpy --cov-report xml --asyncio-mode auto

precommit: isort reformat typecheck lintcheck
