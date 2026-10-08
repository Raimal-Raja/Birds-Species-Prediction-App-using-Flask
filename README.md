# Birds-Species-Prediction-App-using-Flask

Flask image-classification demo that loads a Keras model and predicts among six bird species.

## Repository guide

### Contents

- [Procfile](Procfile)
- [README.md](README.md)
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

Reviewed on 2026-10-08. Python syntax checks passed for 4 source files. Syntax validation does not establish runtime correctness or dependency compatibility.

### Contributions

Describe the issue, reproduction steps, environment, and expected behavior when proposing a change. Keep generated environments, credentials, and unnecessary build artifacts out of new commits.

### License

No top-level license file was found during this review.
