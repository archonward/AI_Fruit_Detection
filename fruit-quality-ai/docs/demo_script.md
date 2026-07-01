# Fruit Quality AI Demo Script

## 1. Introduce the Project

"This project is called Fruit Quality AI. It is a computer vision system that predicts fruit quality from an uploaded image. The project includes dataset preparation, model training, model comparison, real-world testing, error analysis, and a Streamlit demo app."

## 2. Explain the Dataset

"The model was trained using a prepared fruit quality image dataset. The dataset was organized for machine learning so that images could be used for training, validation, and testing."

"I also tested the models on real-world images because I wanted to see whether the model could handle images outside the original dataset."

## 3. Explain Model Training Briefly

"For the modelling approach, I used transfer learning. I compared MobileNetV2 and EfficientNetB0, which are pretrained image classification models."

"Transfer learning was useful because these models already understand general visual patterns such as edges, colors, shapes, and textures. I adapted them to the fruit quality classification task instead of training a deep model completely from scratch."

## 4. Show the Streamlit App

"Now I will open the demo app. The app allows a user to upload a fruit image and receive a predicted class with a confidence score."

Run this command locally:

```bash
streamlit run app/streamlit_app.py
```

## 5. Upload a Sample Image

"I will upload a sample fruit image into the app."

"After the image is uploaded, the app sends it through the selected trained model and displays the prediction."

## 6. Explain Predicted Class and Confidence

"The predicted class is the model's best estimate for the fruit quality category. The confidence score shows how strongly the model selected that class compared with the other possible classes."

"A high confidence score means the model is more certain about the prediction, but it does not guarantee the prediction is correct. A lower confidence score means the image may be harder for the model to classify."

## 7. Explain Limitations

"One important limitation is that the model performs better on images similar to the training dataset. Real-world images can have different lighting, backgrounds, camera angles, blur, or shadows."

"This is called domain shift. In this project, domain shift caused the accuracy to drop when the models were tested on real-world images."

## 8. Explain Model Comparison

"I compared MobileNetV2 and EfficientNetB0."

| Model | Original Test Accuracy | Real-World Accuracy | Notes |
|---|---:|---:|---|
| MobileNetV2 | 97.50% | 81.82% | Selected deployment model |
| EfficientNetB0 | 97.50% | 79.55% | Slightly lower real-world accuracy |

"Both models achieved the same original test accuracy of 97.50%. However, MobileNetV2 performed better on real-world images, achieving 81.82% compared with EfficientNetB0 at 79.55%. Because the demo app should work better on practical uploaded images, I selected MobileNetV2 for deployment."

## 9. End With Future Improvements

"The next step would be to improve real-world robustness. I would collect more real-world fruit images, use stronger data augmentation, test with more difficult images, and improve the app so it can warn users when confidence is low."

"The main lesson from this project is that high test accuracy is not enough by itself. Real-world testing is important because it shows how the model behaves under practical conditions."
