import cv2
import numpy as np

# Function to classify color
def classify_color(hsv_value):
    h, s, v = hsv_value
    if (0 <= h <= 10) or (160 <= h <= 180):
        return "Hot"  # Red
    elif 100 <= h <= 140:
        return "Cold"  # Blue
    elif 35 <= h <= 85:
        return "Natural"  # Green
    elif 20 <= h <= 30:
        return "Warm"  # Yellow
    else:
        return "Unknown"

# Capture image from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the image from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Calculate the histogram to find the most predominant color
    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    predominant_color_hue = np.argmax(hist)

    # Classify the color
    classification = classify_color((predominant_color_hue, 255, 255))

    # Display the classification on the image
    cv2.putText(frame, f"Classification: {classification}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
