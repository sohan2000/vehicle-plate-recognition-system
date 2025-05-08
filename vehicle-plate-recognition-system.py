import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import string
import os
import glob
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import kagglehub

# Configuration
CHAR_IMG_WIDTH = 32
CHAR_IMG_HEIGHT = 32
CHAR_CHANNELS = 1  # Grayscale
MODEL_SAVE_PATH = 'lpr_char_recognition_model.keras'
ALLOWED_CHARACTERS = sorted(list(string.ascii_uppercase + string.digits))
NUM_CLASSES = len(ALLOWED_CHARACTERS)

def load_dataset():
    """Loads the Indian Number Plates dataset"""
    print("\n=== Loading Dataset ===")
    # Download dataset using kagglehub
    path = kagglehub.dataset_download("dataclusterlabs/indian-number-plates-dataset")
    print(f"Dataset downloaded to: {path}")
    
    # Set up paths
    images_folder = os.path.join(path, 'number_plate_images_ocr', 'number_plate_images_ocr')
    annotations_folder = os.path.join(path, 'number_plate_annos_ocr', 'number_plate_annos_ocr')
    
    print(f"Images folder: {images_folder}")
    print(f"Annotations folder: {annotations_folder}")
    
    return images_folder, annotations_folder

def parse_annotation(xml_file_path):
    """Parses XML annotation file to get plate information"""
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        plate_info = {}
        obj = root.find('object')
        if obj is not None:
            plate_info['text'] = obj.find('name').text.upper().replace(" ", "")
            bndbox = obj.find('bndbox')
            plate_info['xmin'] = int(float(bndbox.find('xmin').text))
            plate_info['ymin'] = int(float(bndbox.find('ymin').text))
            plate_info['xmax'] = int(float(bndbox.find('xmax').text))
            plate_info['ymax'] = int(float(bndbox.find('ymax').text))
            return plate_info
        return None
    except Exception as e:
        print(f"Error parsing XML file {xml_file_path}: {e}")
        return None

def load_dataset_info(img_folder_path, ann_folder_path):
    """Loads dataset information including images and annotations"""
    print("\n=== Loading Dataset Information ===")
    dataset = []
    image_files = glob.glob(os.path.join(img_folder_path, '*.jpg')) + \
                 glob.glob(os.path.join(img_folder_path, '*.png'))
    
    print(f"Found {len(image_files)} images")
    
    for img_path in image_files:
        img_filename = os.path.basename(img_path)
        ann_filename = os.path.splitext(img_filename)[0] + '.xml'
        ann_path = os.path.join(ann_folder_path, ann_filename)
        
        if os.path.exists(ann_path):
            annotation = parse_annotation(ann_path)
            if annotation:
                dataset.append({'image_path': img_path, 'annotation': annotation})
    
    print(f"Loaded {len(dataset)} image-annotation pairs")
    return dataset

def extract_plate_roi(image_path, annotation):
    """Extracts the license plate region from the image"""
    print(f"\nExtracting plate ROI from: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    xmin, ymin, xmax, ymax = annotation['xmin'], annotation['ymin'], annotation['xmax'], annotation['ymax']
    plate_roi = image[ymin:ymax, xmin:xmax]
    
    print(f"Plate ROI shape: {plate_roi.shape}")
    return plate_roi

def build_cnn_model(input_shape, num_classes):
    """Builds the CNN model for character recognition"""
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def preprocess_plate(plate_img):
    """Preprocesses the license plate image for character segmentation"""
    # Convert to grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 7
    )
    
    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return cleaned

def segment_characters(plate_img):
    """Segments characters from the license plate"""
    # Preprocess plate
    binary_plate = preprocess_plate(plate_img)
    
    # Find contours
    contours, _ = cv2.findContours(
        binary_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Filter and sort character contours
    char_contours = []
    plate_h, plate_w = plate_img.shape[:2]
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(contour)
        
        # Adjusted filtering criteria
        if (0.1 < aspect_ratio < 2.0 and  # More lenient aspect ratio
            plate_h * 0.2 < h < plate_h * 0.95 and  # More lenient height
            plate_w * 0.02 < w < plate_w * 0.3 and  # More lenient width
            area > (plate_h * plate_w * 0.001)):  # Minimum area threshold
            char_contours.append((x, y, w, h))
    
    # Sort contours from left to right
    char_contours.sort(key=lambda c: c[0])
    
    # If no characters found, try alternative segmentation
    if not char_contours:
        # Try fixed-width segmentation as fallback
        num_chars = 10  # Typical number of characters in a plate
        char_width = plate_w // num_chars
        
        for i in range(num_chars):
            x_start = i * char_width
            x_end = (i + 1) * char_width
            char_img = binary_plate[:, x_start:x_end]
            resized_char = cv2.resize(char_img, (CHAR_IMG_WIDTH, CHAR_IMG_HEIGHT))
            char_contours.append((x_start, 0, char_width, plate_h))
    
    # Extract and resize character images
    char_images = []
    plate_with_boxes = plate_img.copy()
    
    for (x, y, w, h) in char_contours:
        # Draw bounding box
        cv2.rectangle(plate_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Extract and process character
        char_img = binary_plate[y:y+h, x:x+w]
        
        # Add padding to make it square before resizing
        size = max(w, h)
        padded_char = np.zeros((size, size), dtype=np.uint8)
        x_offset = (size - w) // 2
        y_offset = (size - h) // 2
        padded_char[y_offset:y_offset+h, x_offset:x_offset+w] = char_img
        
        # Resize to target dimensions
        resized_char = cv2.resize(padded_char, (CHAR_IMG_WIDTH, CHAR_IMG_HEIGHT))
        char_images.append(resized_char)
    
    # Visualize segmentation results
    plt.figure(figsize=(15, 5))
    
    # Original plate
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))
    plt.title('Original Plate')
    plt.axis('off')
    
    # Binary plate
    plt.subplot(132)
    plt.imshow(binary_plate, cmap='gray')
    plt.title('Binary Plate')
    plt.axis('off')
    
    # Plate with character boxes
    plt.subplot(133)
    plt.imshow(cv2.cvtColor(plate_with_boxes, cv2.COLOR_BGR2RGB))
    plt.title('Detected Characters')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Show segmented characters
    if char_images:
        n_chars = len(char_images)
        fig, axes = plt.subplots(1, n_chars, figsize=(n_chars*2, 2))
        if n_chars == 1:
            axes = [axes]
        for ax, char_img in zip(axes, char_images):
            ax.imshow(char_img, cmap='gray')
            ax.axis('off')
        plt.suptitle('Segmented Characters')
        plt.tight_layout()
        plt.show()
    
    return char_images

def prepare_training_data(dataset_info):
    """Prepares training data from the dataset"""
    print("\n=== Preparing Training Data ===")
    char_images = []
    char_labels = []
    processed_plates = 0
    
    for item in dataset_info:
        plate_roi = extract_plate_roi(item['image_path'], item['annotation'])
        if plate_roi is None:
            continue
            
        plate_text = item['annotation']['text']
        print(f"\nProcessing plate with text: {plate_text}")
        
        segmented_chars = segment_characters(plate_roi)
        print(f"Segmented {len(segmented_chars)} characters")
        
        # More lenient matching - allow for some segmentation errors
        if len(segmented_chars) >= len(plate_text) - 2 and len(segmented_chars) <= len(plate_text) + 2:
            for char_img, char_label in zip(segmented_chars, plate_text):
                if char_label in ALLOWED_CHARACTERS:
                    char_images.append(char_img.reshape(CHAR_IMG_HEIGHT, CHAR_IMG_WIDTH, CHAR_CHANNELS))
                    char_labels.append(char_label)
            processed_plates += 1
    
    print(f"\nSuccessfully processed {processed_plates} plates")
    print(f"Total characters extracted: {len(char_images)}")
    
    if len(char_images) == 0:
        print("ERROR: No characters were extracted. Check segmentation logic and dataset.")
        return None, None, None
    
    # Convert to numpy arrays and normalize
    char_images = np.array(char_images, dtype="float32") / 255.0
    char_labels = np.array(char_labels)
    
    # Encode labels
    label_encoder = LabelEncoder()
    # Fit the encoder on all possible characters first
    label_encoder.fit(ALLOWED_CHARACTERS)
    integer_encoded_labels = label_encoder.transform(char_labels)
    
    # Print unique characters found
    unique_chars = np.unique(char_labels)
    print(f"\nUnique characters in dataset: {''.join(unique_chars)}")
    print(f"Number of unique characters in dataset: {len(unique_chars)}")
    print(f"Total possible characters: {len(ALLOWED_CHARACTERS)}")
    
    # Create one-hot encoded labels with all 36 classes
    onehot_encoded_labels = np.zeros((len(integer_encoded_labels), NUM_CLASSES))
    for i, label in enumerate(integer_encoded_labels):
        onehot_encoded_labels[i, label] = 1
    
    return char_images, onehot_encoded_labels, label_encoder

def train_model(X_train, y_train, X_val, y_val):
    """Trains the character recognition model"""
    model = build_cnn_model(
        input_shape=(CHAR_IMG_HEIGHT, CHAR_IMG_WIDTH, CHAR_CHANNELS),
        num_classes=NUM_CLASSES  # Use all 36 possible characters
    )
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2]
    )
    datagen.fit(X_train)
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True)
    ]
    
    # Train model
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def recognize_plate(image_path, model, label_encoder):
    """Performs end-to-end license plate recognition"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    # For demo purposes, assume plate location is known
    plate_roi = image  # Replace with actual plate detection
    
    # Show original image
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    plt.show()
    
    # Segment characters
    char_images = segment_characters(plate_roi)
    if not char_images:
        print("No characters segmented from plate")
        return None
    
    # Prepare characters for prediction
    chars_to_predict = []
    for char_img in char_images:
        processed_char = char_img.reshape(CHAR_IMG_HEIGHT, CHAR_IMG_WIDTH, CHAR_CHANNELS)
        processed_char = processed_char.astype("float32") / 255.0
        chars_to_predict.append(processed_char)
    
    # Predict characters
    chars_to_predict = np.array(chars_to_predict)
    predictions = model.predict(chars_to_predict)
    predicted_indices = np.argmax(predictions, axis=1)
    predicted_chars = label_encoder.inverse_transform(predicted_indices)
    
    # Combine predictions
    plate_text = "".join(predicted_chars)
    print(f"\nRecognized Plate Text: {plate_text}")
    
    # Draw results on image
    result_image = image.copy()
    cv2.putText(result_image, plate_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show final result
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title('Recognition Result')
    plt.axis('off')
    plt.show()
    
    return plate_text, result_image

def visualize_results(image_path, annotation, predicted_text):
    """Visualizes the recognition results"""
    image = cv2.imread(image_path)
    xmin, ymin, xmax, ymax = annotation['xmin'], annotation['ymin'], annotation['xmax'], annotation['ymax']
    
    # Draw bounding box
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    
    # Draw text
    cv2.putText(image, predicted_text, (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Convert to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    plt.title(f"Actual: {annotation['text']}\nPredicted: {predicted_text}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    print("=== License Plate Recognition System ===")
    print(f"Using TensorFlow version: {tf.__version__}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Allowed characters: {''.join(ALLOWED_CHARACTERS)} (Total: {NUM_CLASSES})")
    
    # 1. Load dataset
    images_folder, annotations_folder = load_dataset()
    
    # 2. Load dataset information
    dataset_info = load_dataset_info(images_folder, annotations_folder)
    
    # 3. Prepare training data
    X, y, label_encoder = prepare_training_data(dataset_info)
    
    if X is None or y is None:
        print("ERROR: Failed to prepare training data. Exiting.")
        exit()
    
    # 4. Split data into training and validation sets
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("\n=== Training Data Shapes ===")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"Number of classes: {NUM_CLASSES}")
    
    # 5. Train model
    print("\n=== Training Model ===")
    model, history = train_model(X_train, y_train, X_val, y_val)
    
    # 6. Test on a few samples
    print("\n=== Testing Model ===")
    num_test_samples = min(5, len(dataset_info))
    for i in range(num_test_samples):
        sample = dataset_info[i]
        print(f"\nProcessing image: {os.path.basename(sample['image_path'])}")
        print(f"Actual plate text: {sample['annotation']['text']}")
        
        plate_roi = extract_plate_roi(sample['image_path'], sample['annotation'])
        if plate_roi is not None:
            char_images = segment_characters(plate_roi)
            if char_images:
                # Prepare characters for prediction
                chars_to_predict = []
                for char_img in char_images:
                    processed_char = char_img.reshape(CHAR_IMG_HEIGHT, CHAR_IMG_WIDTH, CHAR_CHANNELS)
                    processed_char = processed_char.astype("float32") / 255.0
                    chars_to_predict.append(processed_char)
                
                # Predict
                chars_to_predict = np.array(chars_to_predict)
                predictions = model.predict(chars_to_predict)
                predicted_indices = np.argmax(predictions, axis=1)
                predicted_chars = label_encoder.inverse_transform(predicted_indices)
                predicted_text = "".join(predicted_chars)
                
                print(f"Predicted plate text: {predicted_text}")
                
                # Visualize results
                visualize_results(sample['image_path'], sample['annotation'], predicted_text) 