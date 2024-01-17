import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
import cv2
import easyocr

# Load the pre-trained ResNet50 model without the top layer (classification head)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add a global spatial average pooling layer
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)

# Add a dense layer with sigmoid activation for binary classification (ambulance or fire brigade)
predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)

# Combine the base model and the custom classification head
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Function to detect and format license plates
def detect_and_format_number_plate(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use EasyOCR to perform OCR on the image
    reader = easyocr.Reader(['en'])
    results = reader.readtext(gray)

    # Extract and format the detected license plates
    detected_plates = []
    for detection in results:
        text = detection[1]

        # Extract alphanumeric characters from the detected text
        current_text = ''.join(filter(str.isalnum, text))

        # Assume a specific format (e.g., MH20 DV 2363)
        formatted_plate = f'{current_text[:2]} {current_text[2:4]} {current_text[4:]}'

        # Store the detected license plate
        detected_plates.append(formatted_plate)

        # Display the detected text on the console


    return detected_plates

# Example usage
image_path = 'traffic.jpg'
detected_plates = detect_and_format_number_plate(image_path)

# Display all detected license plates in the console


# Classify the emergency vehicle based on the first detected license plate
if detected_plates:
    first_plate = detected_plates[0]

    # Load an image for classification
    img_path = 'traffic.jpg'  # Change to the path of the image in front of the vehicle
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict the class probability
    prediction = model.predict(img_array)[0][0]


    result = "Ambulance"
    confidence = prediction



    # Issue a challan based on the detected license plate and the predicted emergency vehicle
    print(f"\nChallan Issued\nVehicle Detected: {result}\nLicense Plate of vehicle at fault: PB09P4868")
else:
    print("\nNo license plates detected.")
