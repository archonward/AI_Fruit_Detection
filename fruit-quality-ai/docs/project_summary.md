# Fruit Quality AI Project Summary

## 1. Project Title

Fruit Quality AI: Fruit Quality Classification Using Transfer Learning

## 2. Problem Statement

This project aims to classify fruit quality from images using deep learning. The goal is to build a model that can predict the quality category of a fruit image and present the result through a simple demo application.

## 3. Real-World Motivation

Fruit quality inspection is important in food handling, retail, and inventory management. Manual inspection can be slow and inconsistent, especially when many items need to be checked. A computer vision system can help support faster and more consistent visual screening.

## 4. Dataset Summary

The project uses a prepared fruit quality image dataset for model training and testing. The workflow includes dataset preparation so images can be organized into machine learning stages such as training, validation, and testing.

The project also includes real-world image evaluation to test whether the trained models can generalize beyond the original dataset.

## 5. Technical Stack

- Python
- TensorFlow and Keras
- Transfer learning
- MobileNetV2
- EfficientNetB0
- Streamlit
- Image classification workflow
- Dataset preparation pipeline
- Error analysis

## 6. Model Results Table

| Model | Original Test Accuracy | Real-World Accuracy | Deployment Decision |
|---|---:|---:|---|
| MobileNetV2 | 97.50% | 81.82% | Selected deployment model |
| EfficientNetB0 | 97.50% | 79.55% | Comparison model |

## 7. Real-World Testing Summary

Both MobileNetV2 and EfficientNetB0 achieved 97.50% accuracy on the original test set. However, their performance decreased on real-world images.

MobileNetV2 achieved 81.82% real-world accuracy, while EfficientNetB0 achieved 79.55%. This showed that both models were affected by domain shift, where real-world images differ from the original dataset in areas such as lighting, background, camera angle, and image quality.

MobileNetV2 was selected because it had the stronger real-world result.

## 8. Key Lessons Learned

- Transfer learning is effective for image classification projects with limited data and resources.
- Original test accuracy does not always reflect real-world performance.
- Real-world testing is important for understanding practical model reliability.
- Domain shift can significantly reduce accuracy when input images differ from the training data.
- Model selection should consider deployment conditions, not only benchmark results.

## 9. Limitations

- Real-world accuracy is lower than original test accuracy.
- The model may be sensitive to lighting, background clutter, blur, and camera angle.
- The project should not be treated as a perfect replacement for human inspection.
- Confidence scores should be interpreted carefully because high confidence does not always guarantee correctness.
- The known results only compare MobileNetV2 and EfficientNetB0 using the available project evaluation.

## 10. Future Improvements

- Collect more real-world fruit images.
- Add stronger data augmentation for lighting, rotation, blur, and background variation.
- Expand real-world testing with more diverse examples.
- Improve error analysis by grouping incorrect predictions by cause.
- Add confidence thresholds for uncertain predictions.
- Improve the Streamlit interface with clearer feedback for users.
- Test additional model architectures if more data becomes available.
