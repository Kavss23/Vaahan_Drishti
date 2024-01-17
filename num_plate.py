import cv2
import easyocr


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
        print("Detected License Plate:", formatted_plate)

    return detected_plates


# Example usage
image_path = 'car.jpg'
detected_plates = detect_and_format_number_plate(image_path)

# Display all detected license plates in the console
print("\nDetected License Plates:")
for plate in detected_plates:
    print(plate)






