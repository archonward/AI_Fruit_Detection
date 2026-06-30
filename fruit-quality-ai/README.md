# Fruit Quality Classification using Transfer Learning

## Project Summary

This project is a computer vision system for classifying fruit images into fresh
and rotten quality categories. It uses transfer learning to identify both the
fruit type and its freshness condition, such as `fresh_apple`, `rotten_banana`,
or `fresh_orange`.

Fruit quality classification is useful because food inspection is a repeated,
visual task in supermarkets, warehouses, supply chains, and sorting workflows.
Automating part of this process can support faster screening, reduce manual
inspection effort, and help identify poor-quality produce earlier.

What makes this project realistic is that it does not stop at controlled test
accuracy. A separate phone-captured real-world test set was created to measure
how well the models generalise outside the original dataset environment.

## Real-World Motivation

Food quality inspection is important in retail and supply chain operations.
Supermarkets and warehouses need to identify damaged or rotten produce before it
reaches customers, while sorting systems can benefit from fast visual screening.

Better fruit quality detection can also support food waste reduction. If produce
is classified more consistently, businesses can separate sellable, damaged, and
spoiled items earlier in the workflow.

Real-world testing matters because controlled dataset performance can be
misleading. Phone-captured images introduce domain shift through different
lighting, backgrounds, camera angles, distances, and fruit appearances. Testing
on these images gives a more honest view of deployment readiness.

## Dataset

This project uses the **Mendeley Fresh and Rotten Fruits Dataset** as the main
training and evaluation dataset.

- 16 classes
- 3,200 original images
- Balanced dataset with 200 images per class
- Folder-based labels for fresh and rotten fruit categories
- Separate real-world phone-captured test set for generalisation testing
- Dataset files are stored locally and are not committed to GitHub
- Trained model files are stored locally and are not committed to GitHub

Expected local folder structure:

```text
dataset/
  raw/
  processed/
  splits/
    train/
    validation/
    test/
  real_world/
```

## Project Workflow

1. Dataset inspection with `src/analyze_dataset.py`
2. Dataset formatting with `src/format_dataset.py`
3. Train/validation/test split with `src/split_dataset.py`
4. Baseline MobileNetV2 training with `src/train_baseline.py`
5. Model evaluation with `src/evaluate_model.py`
6. Single-image prediction with `src/predict_image.py`
7. Streamlit demo with `app/streamlit_app.py`
8. Real-world testing with `src/evaluate_real_world.py`
9. Model comparison with `src/compare_models.py`

## Model Training

The main baseline model uses **MobileNetV2 transfer learning** with
TensorFlow/Keras.

- Image size: `224 x 224`
- Transfer learning with an ImageNet-pretrained MobileNetV2 backbone
- Frozen backbone during baseline training
- Custom classification head for 16 fruit quality classes
- Data augmentation during training
- Trained for 10 epochs
- Saved model path: `models/baseline_mobilenetv2.keras`

EfficientNetB0 was also trained as a comparison model using the same image size
and dataset split. It reached similar controlled test accuracy but slightly
weaker real-world accuracy.

## Results

| Model | Original Test Accuracy | Real-World Accuracy | Notes |
| --- | ---: | ---: | --- |
| MobileNetV2 | 97.50% | 81.82% | Preferred deployment model |
| EfficientNetB0 | 97.50% | 79.55% | Similar test accuracy, weaker real-world generalisation |

Additional EfficientNetB0 metrics:

- Train accuracy: `98.18%`
- Validation accuracy: `97.08%`
- Test loss: `0.1195`

## Key Findings

- High test accuracy does not guarantee real-world performance.
- Real-world phone images introduced domain shift.
- Error analysis helped identify weak classes.
- Rotten apple was a weak class in real-world testing.
- Targeted retraining improved rotten apple performance.
- MobileNetV2 was retained because it gave better real-world performance while
  remaining lightweight.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train MobileNetV2:

```bash
python src/train_baseline.py
```

Evaluate MobileNetV2:

```bash
python src/evaluate_model.py
```

Predict a single image:

```bash
python src/predict_image.py path/to/your_image.jpg
```

Run the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

Train EfficientNetB0:

```bash
python src/train_efficientnet.py
```

Compare models:

```bash
python src/compare_models.py
```

## Project Structure

```text
fruit-quality-ai/
|-- app/
|   `-- streamlit_app.py
|-- dataset/
|   |-- raw/
|   |-- processed/
|   |-- splits/
|   `-- real_world/
|-- models/
|   |-- baseline_mobilenetv2.keras
|   `-- efficientnetb0.keras
|-- outputs/
|   |-- figures/
|   `-- reports/
|-- src/
|   |-- analyze_dataset.py
|   |-- analyze_real_world_errors.py
|   |-- compare_models.py
|   |-- config.py
|   |-- evaluate_model.py
|   |-- evaluate_real_world.py
|   |-- evaluate_real_world_efficientnet.py
|   |-- format_dataset.py
|   |-- predict_image.py
|   |-- split_dataset.py
|   |-- train_baseline.py
|   `-- train_efficientnet.py
|-- requirements.txt
`-- README.md
```

## Limitations

- The real-world dataset is still small.
- Rotten fruit variation is limited.
- The original dataset is controlled and may not represent all deployment
  conditions.
- The system performs image classification only; it does not include object
  detection yet.
- The model may struggle with unusual lighting, backgrounds, camera angles, or
  highly ambiguous fruit conditions.

## Future Improvements

- Collect a larger real-world dataset.
- Add more fruit types and quality conditions.
- Fine-tune selected layers of the transfer learning backbone.
- Try a two-stage classifier: fruit type first, freshness second.
- Deploy the Streamlit app online.
- Add explainability using Grad-CAM.
