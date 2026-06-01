# Fruit Quality AI

Fruit Quality AI is a beginner-friendly computer vision project for classifying fruit images by quality, such as fresh or rotten.

## Project Goal

The goal of this project is to build an AI image classification workflow that can inspect fruit images, prepare the data, train a baseline model, evaluate performance, and later run simple predictions on new images.

## Planned Dataset

This project will later use a Mendeley fresh/rotten fruits dataset.

Dataset files are not included yet and should not be committed to GitHub. Keep downloaded dataset files inside the local `dataset/` folder only.

## Planned Workflow

1. Dataset inspection
2. Preprocessing
3. Train/validation/test split
4. Baseline model training
5. Model evaluation
6. Simple prediction demo

## Project Structure

```text
fruit-quality-ai/
├── dataset/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── notebooks/
├── src/
├── models/
├── outputs/
│   ├── figures/
│   └── reports/
├── requirements.txt
├── README.md
└── .gitignore
```

## Getting Started

Install the project dependencies with:

```bash
pip install -r requirements.txt
```

After adding the dataset locally, inspect the raw class folders with:

```bash
python src/analyze_dataset.py
```
