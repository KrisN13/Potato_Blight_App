# Potato Disease Classification App

This project builds and deploys a convolutional neural network that classifies potato leaf images into three categories: Early Blight, Late Blight and Healthy. The model was trained using TensorFlow and deployed as a serverless API on Google Cloud Functions. A simple web interface allows users to upload an image and receive predictions in real time.

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

1. Install Python dependencies
2. Place the exported model under a `/models` directory
3. Start the local Functions Framework
4. Send a POST request with an image: curl -X POST -F "file=@your_image.jpg"


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