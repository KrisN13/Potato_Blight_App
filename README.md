# Potato Disease Classification App

## Snapshot

- Application detects Early, and Late Blight, as well as healthy potato plant leaves
- Simple web interface allows users to upload an image and receive predictions in real time
- Achieved a +90% classification accuracy
- Over 2,500 images used to train the model
- Trained using TensorFlow
- Deployed as a serverless API on Google Cloud
- Primarily aimed at farmers and agriculturists who are trying to maintain healthy potato plants

## Problem

Potato blight is a fungal disease responsible for significant crop losses globally, including the Irish Famine of the 1840s. Early detection is critical, as late-stage blight can destroy an entire crop within days. Traditional identification relies on expert visual inspection, which is inaccessible to many smallholder farmers.

## Project structure

The repository includes:

- Notebook(s) for data loading, preprocessing and training the CNN model
- Exported TensorFlow SavedModel files
- Cloud Function code for serving predictions through an HTTP endpoint
- Frontend HTML, CSS and JavaScript used to build the upload-and-classify interface
- Example inference requests

## Model

The model is a convolutional neural network trained on the PlantVillage potato leaf dataset. Images were resized to 256×256 pixels and kept in the 0 to 255 range as float32 values. The model outputs a softmax probability across the three classes:

1. Early Blight
2. Late Blight
3. Healthy

This class order is used consistently across training and inference.

<a href="https://krisnoondata.com/potato-blight-disease-application">
  <img src="https://github.com/KrisN13/Potato_Blight_App/blob/main/images/Potato_App_Interface.png" alt="Application Interface" width="700"/>
</a>

*Upload a potato leaf image to receive an instant Early Blight, Late Blight, or Healthy classification.*

| Class | Accuracy |
|-------|----------|
| Early Blight | 92%+ |
| Late Blight | 90%+ |
| Healthy | 90%+ |

## Deployment

The trained model is deployed using Google Cloud Functions. The function:

- Loads the SavedModel using `tf.saved_model.load`
- Accepts an uploaded image via HTTP POST
- Applies the same preprocessing used during training
- Returns the predicted class and confidence score as JSON
- Restricts CORS access to the production website

The model files are stored in a Google Cloud Storage bucket and downloaded on cold starts.

## Web application

The web interface is a lightweight HTML page that allows users to:

- Select an image
- Preview the uploaded file
- Submit the image for classification
- View the returned prediction and confidence

The interface enforces a 15 MB upload limit and allows one classification per image. The layout is responsive and minimal, styled to fit into a WordPress Gutenberg block.

## How to run locally

1. Install Python dependencies: `pip install -r requirements.txt`
2. The trained model is loaded automatically from Google Cloud Storage on startup — no local model files required
3. Start the local Functions Framework
4. Send a POST request with an image: `curl -X POST -F "file=@your_image.jpg"`

## Notes

- The SavedModel format is loaded using TensorFlow's low-level SavedModel APIs, which is required in Keras 3
- Model accuracy depends heavily on preprocessing consistency
- Prediction confidence may be affected by low-quality or overly compressed images

## Future improvements

Possible extensions of the project include:

- Expanding to additional crop diseases
- Improving robustness with more varied real-world images
- Building a mobile-friendly TensorFlow Lite version
- Creating a batch-processing API endpoint
