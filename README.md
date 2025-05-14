## Vehicle Plate Recognition System

This project is a deep learning-based system for automatic vehicle license plate recognition, focused on Indian number plates. It uses OpenCV for image processing and TensorFlow/Keras for character recognition.

**Features**
- Downloads and processes the Indian Number Plates dataset.
- Detects and segments license plate regions from images.
- Segments individual characters from license plates using image processing.
- Trains a Convolutional Neural Network (CNN) to recognize alphanumeric characters.
- Provides end-to-end plate recognition and result visualization.

**Requirements**
- Python 3.x
- TensorFlow
- OpenCV
- NumPy
- scikit-learn
- matplotlib
- kagglehub

**How to Use**
1. Make sure all dependencies are installed.
2. Run the script. It will:
   - Download the dataset automatically.
   - Prepare and preprocess the data.
   - Train a CNN model for character recognition.
   - Test the recognition pipeline on sample images and visualize results.

**Notes**
- The model is tailored for uppercase English letters and digits (A-Z, 0-9).
- For best results, use the provided dataset structure.

**Project Structure**
- Data loading and annotation parsing
- Plate region extraction and character segmentation
- Model building and training
- End-to-end plate recognition and visualization

**License**
This project is for educational and research purposes. Please check dataset and dependency licenses before commercial use.
