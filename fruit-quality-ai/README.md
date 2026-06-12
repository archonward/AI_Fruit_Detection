# Fruit Quality AI

## Project Overview

Fruit Quality AI is a computer vision project for classifying fruit images into
16 freshness and quality categories, such as fresh apple or rotten banana. The
project includes dataset preparation, train/validation/test splitting, baseline
model training, evaluation, single-image prediction, and a Streamlit demo app
for portfolio presentation.

## Real-World Motivation

Fruit quality inspection matters in grocery retail, food supply chains, sorting
systems, and waste reduction workflows. Manual inspection is time-consuming and
can be inconsistent. A lightweight image classification pipeline can help show
how machine learning might support early quality screening in a structured
environment.

## Dataset

This project uses the **Mendeley Fresh and Rotten Fruits Dataset**.

- 16 classes
- Fresh and rotten fruit categories
- Image classification setup using folder-based labels
- Local dataset storage only; dataset files should not be committed

Expected local folder structure:

```text
dataset/
  raw/
  processed/
  splits/
    train/
    validation/
    test/
```

## Project Workflow

1. Inspect the dataset structure with `src/analyze_dataset.py`
2. Format or organize raw data with `src/format_dataset.py`
3. Create train, validation, and test splits with `src/split_dataset.py`
4. Train the baseline MobileNetV2 model with `src/train_baseline.py`
5. Evaluate the saved model with `src/evaluate_model.py`
6. Run single-image inference with `src/predict_image.py`
7. Present the trained model in the Streamlit demo app at `app/streamlit_app.py`

## Model Approach

The baseline model uses **MobileNetV2 transfer learning** with TensorFlow/Keras.

- Input size: `224 x 224`
- Base architecture: `MobileNetV2`
- Transfer learning with ImageNet weights
- Data augmentation during training
- Final classifier head for 16 classes
- Saved model path: `models/baseline_mobilenetv2.keras`

The current Streamlit demo follows the same preprocessing logic as the working
single-image prediction script:

- convert image to RGB
- resize to `224 x 224`
- convert to NumPy array
- add batch dimension
- keep the raw `0-255` pixel range
- do not apply `mobilenet_v2.preprocess_input` again during inference

## Results

Baseline model results:

- Train accuracy: `98.48%`
- Validation accuracy: `97.29%`
- Test accuracy: `97.50%`
- Test loss: `0.1149`

These results indicate strong performance on the controlled dataset split, with
good generalisation from training to validation and test data.

## Project Structure

```text
fruit-quality-ai/
|-- app/
|   `-- streamlit_app.py
|-- dataset/
|   |-- raw/
|   |-- processed/
|   `-- splits/
|       |-- train/
|       |-- validation/
|       `-- test/
|-- models/
|   `-- baseline_mobilenetv2.keras
|-- notebooks/
|-- outputs/
|   |-- figures/
|   `-- reports/
|-- src/
|   |-- analyze_dataset.py
|   |-- config.py
|   |-- evaluate_model.py
|   |-- format_dataset.py
|   |-- predict_image.py
|   |-- split_dataset.py
|   `-- train_baseline.py
|-- requirements.txt
`-- README.md
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## How To Run Training

Before training, make sure the processed dataset exists and the split folders
have been created.

Create dataset splits:

```bash
python src/split_dataset.py
```

Train the baseline model:

```bash
python src/train_baseline.py
```

Training saves:

- model file in `models/`
- training plots in `outputs/figures/`
- training report in `outputs/reports/`

## How To Run Evaluation

Evaluate the saved model on the test split:

```bash
python src/evaluate_model.py
```

Evaluation saves:

- confusion matrix
- normalised confusion matrix
- misclassified examples figure
- classification report
- evaluation summary
- test predictions CSV

## How To Run Single-Image Prediction

Run single-image inference with:

```bash
python src/predict_image.py path/to/your_image.jpg
```

Optional debug mode:

```bash
python src/predict_image.py path/to/your_image.jpg --debug
```

The script prints:

- image path
- class list
- predicted class
- confidence score
- top 3 predictions

## How To Run The Streamlit Demo App

Launch the demo app locally with:

```bash
streamlit run app/streamlit_app.py
```

The app allows you to:

- upload a fruit image
- preview the uploaded image
- run prediction with the trained model
- view the predicted class and confidence score
- inspect the top 3 predicted classes

## Limitations

- The model was trained on a controlled dataset.
- Real-world performance may vary depending on lighting, background, camera
  angle, image quality, and fruit condition.
- The current system is a classification demo, not a full production inspection
  pipeline.
- The project currently focuses on one baseline architecture.

## Future Improvements

- add more diverse real-world images for stronger robustness
- compare multiple transfer learning backbones
- add fine-tuning after baseline transfer learning
- improve handling of ambiguous or low-confidence cases
- package the app for easier deployment
- add model explainability features such as saliency or Grad-CAM
