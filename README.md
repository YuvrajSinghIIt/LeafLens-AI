# 🌿 LeafLens AI — Tomato Leaf Disease Detection

**EC-422 Artificial Intelligence course project** | Deep Learning | Computer Vision | Transfer Learning

LeafLens AI is an interactive Streamlit application for preliminary classification of tomato leaf images into five categories: **Healthy, Bacterial Spot, Early Blight, Late Blight, and Leaf Mold**.

## Features

- Image upload and real neural-network predictions
- Confidence distribution over five classes
- Interactive evaluation dashboard with accuracy, F1 and confusion matrix
- Educational disease reference and presentation walkthrough
- MobileNetV3-Small transfer learning with PyTorch

## Evaluation

The trained experiment (500 images; train 350, validation 75, test 75) achieved **62/75 = 82.67% test accuracy**, macro **F1 ≈ 0.831**, across 8 epochs. These results are specific to the held-out dataset images, not unrestricted field photographs. Do not use for treatment decisions without agricultural expertise.

## Run locally

Requires Python 3.10–3.12:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501.

## Render deployment

Use **Python** runtime, **Free** plan (subject to RAM limitations), Singapore region:

- Build: `bash render-build.sh`
- Start: `bash render-start.sh`
- Environment: `PYTHON_VERSION=3.11.11`

The trained `models/leaflens_tomato.pt` checkpoint must be present in the repository before deployment. Free-tier RAM may not be sufficient for PyTorch.

## Training

Model: PyTorch TorchVision **MobileNetV3-Small**. Dataset: [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset). Training and evaluation workflows are included in the full project bundle.

## Limitations

Course demonstration only. The model covers only five tomato-leaf categories; environmental variation and unseen diseases can reduce accuracy. Prediction confidence is not certainty.

## Repository status

The Streamlit source, saved checkpoint, real metrics, Render scripts and presentation guide are committed. See `PRESENTATION_GUIDE.md` for the course walkthrough.
