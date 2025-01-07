import cv2
import numpy as np
import tensorflow as tf
import os

# Define the labels for classification
class_labels = ["Organic", "Recyclable"]

# Define the image size for resizing
IMG_SIZE = (224, 224)

# Load the trained model from the directory
model_path = os.path.join("model", "waste_classifier_model.h5")
model = tf.keras.models.load_model(model_path)

# Load the object detection model
prototxt_path = os.path.join("model", "MobileNetSSD_deploy.prototxt")
model_weights_path = os.path.join("model", "MobileNetSSD_deploy.caffemodel")
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_weights_path)

# Open the camera
cap = cv2.VideoCapture(0)

# Create a named window
cv2.namedWindow("Camera")

# Set the window size
cv2.resizeWindow("Camera", 800, 600)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Perform object detection
    blob = cv2.dnn.blobFromImage(frame, 0.007843, IMG_SIZE, (127.5, 127.5, 127.5), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()

    # Process each detected object
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.5:
            # Get the coordinates of the detected object
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype(int)

            # Extract the object from the frame
            object_img = frame[startY:endY, startX:endX]

            # Check if the object image is valid
            if object_img.shape[0] <= 0 or object_img.shape[1] <= 0:
                continue

            # Preprocess the object image
            object_resized = cv2.resize(object_img, IMG_SIZE)
            object_normalized = object_resized / 255.0
            object_expanded = np.expand_dims(object_normalized, axis=0)

            # Make prediction
            predictions = model.predict(object_expanded)
            predicted_class = np.argmax(predictions)
            prediction_percentage = np.max(predictions) * 100

            # Get the predicted class label and percentage
            class_label = class_labels[predicted_class]
            prediction_text = f"{class_label}: {prediction_percentage:.2f}%"

            # Draw a rectangle around the detected object
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # Display the predicted class label and percentage
            cv2.putText(frame, prediction_text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Camera", frame)

    # Capture and save the image when 'c' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('c'):
        cv2.imwrite("captured_image.jpg", frame)
        print("Image captured and saved as 'captured_image.jpg'")
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
