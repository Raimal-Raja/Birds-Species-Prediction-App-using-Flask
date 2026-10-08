# Birds-Species-Prediction-App-using-Flask

Flask image-classification demo that loads a Keras model and predicts among six bird species.

## Setup and repository reference

### Project structure

- [Procfile](Procfile)
- [app](app)
- [app.py](app.py)
- [config.py](config.py)
- [requirements.txt](requirements.txt)
- [sample_data](sample_data)

### Getting started

```bash
git clone https://github.com/Raimal-Raja/Birds-Species-Prediction-App-using-Flask.git
cd Birds-Species-Prediction-App-using-Flask
```

Create and activate a virtual environment, then install the project dependencies:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r "requirements.txt"
```

Application entry point:

```bash
python app.py
```

### Configuration and limitations

Inference requires app/static/model/bird_species.h5 and its compatible Keras/TensorFlow environment. The trained model was not loaded in this review.

### Validation

Audit: 2026-10-08. Repository structure, setup instructions and description were reviewed. 4 existing Python files passed syntax checks; changed files and new regression tests were checked separately. Syntax checks do not establish full runtime correctness. External APIs, live scraping, GUI interaction, notebook training and production deployment were not comprehensively exercised.

### Repository description

The short GitHub description is provided in [REPOSITORY_DESCRIPTION.md](REPOSITORY_DESCRIPTION.md).

### Contributions

Describe the issue, reproduction steps, environment, and expected behavior when proposing a change. Keep generated environments, credentials, and unnecessary build artifacts out of new commits.

### License

No top-level license file was found during this review.
