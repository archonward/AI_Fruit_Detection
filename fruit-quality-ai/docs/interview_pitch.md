# Fruit Quality AI Interview Pitch

## 30-Second Pitch

Fruit Quality AI is a computer vision project that classifies fruit quality from images. The goal is to identify whether fruit appears fresh or defective using deep learning, then make the model usable through a Streamlit demo app. I trained and compared MobileNetV2 and EfficientNetB0 using transfer learning. Both models reached 97.50% accuracy on the original test set, but MobileNetV2 performed better on real-world images with 81.82% accuracy compared with EfficientNetB0 at 79.55%, so I selected MobileNetV2 for deployment.

## 1-Minute Pitch

This project focuses on fruit quality detection using image classification. The problem matters because manual fruit inspection can be slow, inconsistent, and difficult to scale. A computer vision system can support faster quality checking in settings such as retail, food handling, or inventory screening.

The project uses a prepared image dataset for training and testing, then evaluates the trained models on additional real-world images. I used transfer learning with MobileNetV2 and EfficientNetB0 because these pretrained convolutional neural networks already understand useful visual features such as edges, textures, colors, and shapes. This makes training more practical for a student-scale dataset.

Both models achieved 97.50% accuracy on the original test set. However, real-world testing showed a performance drop, which highlighted domain shift: the real photos differed from the training images in lighting, background, camera quality, and fruit appearance. MobileNetV2 achieved 81.82% real-world accuracy, while EfficientNetB0 achieved 79.55%, so MobileNetV2 was chosen as the deployment model.

## Detailed Technical Explanation

### 1. What the Project Is

Fruit Quality AI is an image classification project that predicts fruit quality from uploaded images. It includes a dataset preparation pipeline, model training, model comparison, real-world image evaluation, error analysis, and a Streamlit demo application.

The project is designed as an end-to-end university portfolio project. It does not stop at training accuracy; it also checks how well the model performs on images that are closer to real user inputs.

### 2. Why the Problem Matters

Fruit quality inspection is commonly done by people, which can be time-consuming and inconsistent. Visual defects, ripeness differences, bruising, and freshness changes can affect whether fruit should be sold, stored, or rejected.

A computer vision model can help support this process by giving fast and repeatable predictions. It would not replace expert inspection completely, but it can be useful as a screening tool or decision-support system.

### 3. Dataset Used

The project uses an image dataset prepared for fruit quality classification. The dataset preparation pipeline organizes the images into training, validation, and testing workflows so the models can be trained and measured consistently.

In addition to the original dataset test split, the project includes real-world image evaluation. This is important because a model can perform well on a clean dataset but behave differently when tested with images taken in less controlled conditions.

### 4. Model Approach

The project compares two transfer learning models:

- MobileNetV2
- EfficientNetB0

Both models are convolutional neural network architectures commonly used for image classification. The project trains them on the fruit quality dataset and compares their performance on both the original test set and real-world images.

### 5. Why Transfer Learning Was Used

Transfer learning was used because training a deep neural network from scratch usually requires a very large dataset and significant computing resources. Pretrained models such as MobileNetV2 and EfficientNetB0 have already learned general visual features from large image datasets.

By reusing these learned features and adapting the final classification layers to the fruit quality task, the project can train more efficiently and achieve strong performance with a smaller project dataset.

### 6. How the Model Was Evaluated

The models were evaluated in two main ways:

- Original test set accuracy: measures performance on the held-out test portion of the prepared dataset.
- Real-world image accuracy: measures performance on additional images that better represent practical usage conditions.

The project also includes error analysis to understand where predictions fail and why the model may be less reliable under certain image conditions.

### 7. What Happened During Real-World Testing

Both models performed strongly on the original test set, with each reaching 97.50% accuracy. However, real-world testing showed lower accuracy:

- MobileNetV2: 81.82% real-world accuracy
- EfficientNetB0: 79.55% real-world accuracy

This drop showed that the models had learned useful fruit quality features, but they were also affected by differences between the original dataset and real-world images.

### 8. What Domain Shift Means in This Project

Domain shift means the training and testing environments are different. In this project, the original dataset images may have more consistent lighting, backgrounds, framing, or image quality than real-world photos.

For fruit quality detection, domain shift can happen because of:

- Different lighting conditions
- Different camera angles
- Background clutter
- Different fruit sizes or positions
- Shadows, reflections, or blur
- Natural variation in fruit appearance

The real-world accuracy results show why domain shift matters. A model can look very accurate on the original test set but still lose performance when the input images come from a different environment.

### 9. Why MobileNetV2 Was Selected Over EfficientNetB0

MobileNetV2 and EfficientNetB0 both achieved 97.50% accuracy on the original test set. The deciding factor was real-world performance.

MobileNetV2 achieved 81.82% accuracy on real-world images, while EfficientNetB0 achieved 79.55%. Since the deployment model should perform better on practical user-uploaded images, MobileNetV2 was selected.

This decision was based on real-world evaluation rather than only relying on the original test accuracy.

### 10. What I Would Improve Next

The next improvements would focus on making the model more robust outside the original dataset:

- Collect more real-world fruit images under varied lighting and backgrounds.
- Add stronger data augmentation to simulate real-world conditions.
- Expand the test set with more challenging examples.
- Improve error analysis by grouping mistakes by cause, such as lighting, blur, or background.
- Consider confidence thresholds so the app can flag uncertain predictions.
- Test the model on more fruit types or quality categories if the dataset is expanded.
- Improve the Streamlit app with clearer prediction feedback and user guidance.

Overall, the key improvement would be reducing the gap between dataset performance and real-world performance.
