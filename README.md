# iris-mlops

Small demo MLOps project using the Iris dataset, integrated with GitHub Actions and SonarCloud.

Project structure

iris-mlops/
- .github/workflows/sonarcloud.yml
- src/
  - __init__.py
  - data_processing.py
  - train.py
  - inference.py
- tests/
  - test_pipeline.py
- requirements.txt
- sonar-project.properties
- .gitignore
- README.md

Quickstart (local)

1. Create a Python 3.10 virtual environment and activate it:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

2. Install pinned dependencies:

```bash
pip install -r requirements.txt
```

3. Run the data pipeline and training:

```bash
python src/data_processing.py
python src/train.py
```

4. Run inference with example values:

```bash
python src/inference.py --sepal_length 5.1 --sepal_width 3.5 --petal_length 1.4 --petal_width 0.2
```

5. Run tests with coverage (generates `coverage.xml`):

```bash
coverage run --source=src -m pytest
coverage xml -o coverage.xml
```

Connect to SonarCloud

1. Sign in to SonarCloud and create an organization and project for this repository.
2. In SonarCloud, note the `projectKey` and `organization` and place them into `sonar-project.properties`.
3. Create a GitHub repository and push this project.
4. In the GitHub repository settings, add a secret named `SONAR_TOKEN` with the token from SonarCloud.
5. The provided GitHub Actions workflow `.github/workflows/sonarcloud.yml` will run on push and pull requests, run tests and upload coverage to SonarCloud.

