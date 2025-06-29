// Define all your code snippets here with languages
const codeSnippets = {
  full: {
    code: `import cv2

# Attempt to use the DirectShow backend for capturing video on Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video stream or file")
    exit()

try:
    # Looping continuously to get frames
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly, ret is True
        if not ret:
            print("Error: Failed to capture image")
            break

        # Display the resulting frame
        cv2.imshow('Webcam Video', frame)

        # Break the loop on 'q' key press (waitKey returns a 32-bit integer)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # When everything is done, release the capture
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
    `,
    language: 'python'
  },
  grayscale: {
    code: `# Convert the frame to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    `,
    language: 'python'
  },
  gaussianBlur: {
    code: `# Apply GaussianBlur to reduce noise and improve contour detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    `,
    language: 'python'
  },
  canny: {
    code: `# Detect edges using Canny
edged = cv2.Canny(blurred, 30, 100)  # Adjusted thresholds for Canny
    `,
    language: 'python'
  },
  contourDetect: {
    code: `# Find contours in the edged image
contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    # Filter out small contours
    if cv2.contourArea(contour) < 500:
        continue
    peri = cv2.arcLength(contour, True);
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True);
    `,
    language: 'python'
  },
  shape: {
    code: `# Draw the contour and the name of the shape on the image
    cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2);
    M = cv2.moments(contour);
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"]);
        cY = int(M["m01"] / M["m00"]);
        cv2.putText(frame, shape, (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2);
    `,
    language: 'python'
  }
};

export default codeSnippets;
