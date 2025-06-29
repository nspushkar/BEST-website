const codeSections = {
    Section1: {
      code: `# Import Libraries
  import cv2
  import numpy as np
  import datetime
  import csv
  from sklearn.ensemble import IsolationForest
  import joblib
  import os
  
  # Initialize video capture
  cap = cv2.VideoCapture(0)
  `,
      language: 'python',
    },
    Section2: {
      code: `# Create directory to save anomaly frames
  if not os.path.exists('anomalies'):
      os.makedirs('anomalies')
  
  # Open CSV file for writing anomaly log
  anomaly_log_file = open('anomaly_log.csv', 'w', newline='')
  anomaly_log_writer = csv.writer(anomaly_log_file)
  anomaly_log_writer.writerow(["Timestamp", "ImageFile"])
  
  # Open CSV file for writing data log
  data_log_file = open('data_log.csv', 'w', newline='')
  data_log_writer = csv.writer(data_log_file)
  data_log_writer.writerow(["Timestamp", "Motion"])
  `,
      language: 'python',
    },
    Section3: {
      code: `# Read initial frames
  ret, frame1 = cap.read()
  ret, frame2 = cap.read()
  data_log = []
  
  # Flag to check if model is ready for anomaly detection
  model_ready = False
  `,
      language: 'python',
    },
    Section4: {
      code: `while cap.isOpened():
      # Motion detection
      diff = cv2.absdiff(frame1, frame2)
      gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
      blur = cv2.GaussianBlur(gray, (5, 5), 0)
      _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
      dilated = cv2.dilate(thresh, None, iterations=3)
      contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  
      motion = 0
      for contour in contours:
          if cv2.contourArea(contour) < 900:
              continue
          x, y, w, h = cv2.boundingRect(contour)
          cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
          motion += 1
  
      frame1 = frame2
      ret, frame2 = cap.read()
  
      # Save data with timestamp
      timestamp = datetime.datetime.now()
      data_log_writer.writerow([timestamp, motion])
      data_log.append([motion])
  `,
      language: 'python',
    },
    Section5: {
      code: `# Initial Model Training and Retraining
  if len(data_log) == 100:
      model = IsolationForest(contamination=0.01)
      model.fit(data_log)
      joblib.dump(model, 'isolation_forest_model.pkl')
      print("Initial model training complete. Model is now ready to detect anomalies.")
      print("Select feed window and press q to quit")
      model_ready = True
  
  # Periodic Model Retraining
  if len(data_log) > 100 and len(data_log) % 50 == 0:  # Retrain every 50 new frames
      model = IsolationForest(contamination=0.01)
      model.fit(data_log)
      joblib.dump(model, 'isolation_forest_model.pkl')
      print("Model retrained and updated.")
  `,
      language: 'python',
    },
    Section6: {
      code: `# Anomaly Detection and Logging
  if model_ready:
      feature_vector = np.array([[motion]])
      anomaly = model.predict(feature_vector)
      if anomaly == -1:
          print(f"Anomaly detected at {timestamp}")
          # Save the frame to file
          anomaly_filename = f"anomalies/anomaly_{timestamp.strftime('%Y%m%d_%H%M%S')}.png"
          cv2.imwrite(anomaly_filename, frame1)
          # Log anomaly to CSV file
          anomaly_log_writer.writerow([timestamp, anomaly_filename])
  `,
      language: 'python',
    },
    Section7: {
      code: `# Display Video and Clean Up
  cv2.imshow("feed", frame1)
  if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  
  # Release resources
  cap.release()
  cv2.destroyAllWindows()
  anomaly_log_file.close()
  data_log_file.close()
  print("Video capture released and windows destroyed. Exiting program.")
  `,
      language: 'python',
    },
    full: {
      code: `import cv2
  import numpy as np
  import datetime
  import csv
  from sklearn.ensemble import IsolationForest
  import joblib
  import os
  
  # Initialize video capture
  cap = cv2.VideoCapture(0)
  
  # Create directory to save anomaly frames
  if not os.path.exists('anomalies'):
      os.makedirs('anomalies')
  
  # Open CSV file for writing anomaly log
  anomaly_log_file = open('anomaly_log.csv', 'w', newline='')
  anomaly_log_writer = csv.writer(anomaly_log_file)
  anomaly_log_writer.writerow(["Timestamp", "ImageFile"])
  
  # Open CSV file for writing data log
  data_log_file = open('data_log.csv', 'w', newline='')
  data_log_writer = csv.writer(data_log_file)
  data_log_writer.writerow(["Timestamp", "Motion"])
  
  # Read initial frames
  ret, frame1 = cap.read()
  ret, frame2 = cap.read()
  data_log = []
  
  # Flag to check if model is ready for anomaly detection
  model_ready = False
  
  while cap.isOpened():
      # Motion detection
      diff = cv2.absdiff(frame1, frame2)
      gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
      blur = cv2.GaussianBlur(gray, (5, 5), 0)
      _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
      dilated = cv2.dilate(thresh, None, iterations=3)
      contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  
      motion = 0
      for contour in contours:
          if cv2.contourArea(contour) < 900:
              continue
          x, y, w, h = cv2.boundingRect(contour)
          cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
          motion += 1
  
      frame1 = frame2
      ret, frame2 = cap.read()
  
      # Save data with timestamp
      timestamp = datetime.datetime.now()
      data_log_writer.writerow([timestamp, motion])
      data_log.append([motion])
  
      # Initial model training
      if len(data_log) == 100:
          model = IsolationForest(contamination=0.01)
          model.fit(data_log)
          joblib.dump(model, 'isolation_forest_model.pkl')
          print("Initial model training complete. Model is now ready to detect anomalies.")
          print("Select feed window and press q to quit")
          model_ready = True
  
      # Periodic model retraining
      if len(data_log) > 100 and len(data_log) % 50 == 0:  # Retrain every 50 new frames
          model = IsolationForest(contamination=0.01)
          model.fit(data_log)
          joblib.dump(model, 'isolation_forest_model.pkl')
          print("Model retrained and updated.")
  
      # Anomaly detection
      if model_ready:
          feature_vector = np.array([[motion]])
          anomaly = model.predict(feature_vector)
          if anomaly == -1:
              print(f"Anomaly detected at {timestamp}")
              # Save the frame to file
              anomaly_filename = f"anomalies/anomaly_{timestamp.strftime('%Y%m%d_%H%M%S')}.png"
              cv2.imwrite(anomaly_filename, frame1)
              # Log anomaly to CSV file
              anomaly_log_writer.writerow([timestamp, anomaly_filename])
  
      # Display video
      cv2.imshow("feed", frame1)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
  
  # Release resources
  cap.release()
  cv2.destroyAllWindows()
  anomaly_log_file.close()
  data_log_file.close()
  print("Video capture released and windows destroyed. Exiting program.")
  `,
      language: 'python',
    },
  };
  
  export default codeSections;
  