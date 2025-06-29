const codeSections = {
    full: {
      code: `# Install necessary libraries
  !pip install tensorflow scikit-learn google-cloud-dialogflow
  
  # Import libraries
  import numpy as np
  import tensorflow as tf
  from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
  from tensorflow.keras.applications import VGG16
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import StandardScaler
  from sklearn.metrics.pairwise import cosine_similarity
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.cluster import KMeans
  from sklearn.metrics import classification_report
  import matplotlib.pyplot as plt
  from google.cloud import dialogflow_v2 as dialogflow
  import os
  import pandas as pd
  import json
  
  # Define the path to your dataset
  dataset_path = '/kaggle/input/real-life-industrial-dataset-of-casting-product'
  
  # Load dataset (assuming data is organized in directories by class)
  datagen = ImageDataGenerator(rescale=1./255)
  dataset = datagen.flow_from_directory(dataset_path, target_size=(224, 224), batch_size=32, class_mode='binary')
  
  # Function to load and prepare the data
  def load_data(dataset):
      features = []
      labels = []
      for batch in dataset:
          X_batch, y_batch = batch
          features.extend(X_batch)
          labels.extend(y_batch)
          if len(features) >= dataset.samples:
              break
      return np.array(features), np.array(labels)
  
  X, y = load_data(dataset)
  
  # Split data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
  # Normalize and reshape the features
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train.reshape(-1, 224 * 224 * 3))
  X_test = scaler.transform(X_test.reshape(-1, 224 * 224 * 3))
  
  # Load pre-trained VGG16 model for feature extraction
  base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
  model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
  
  # Function to extract features
  def extract_features(model, data):
      features = model.predict(data)
      return features.reshape(features.shape[0], -1)
  
  X_train_features = extract_features(model, X_train.reshape(-1, 224, 224, 3))
  X_test_features = extract_features(model, X_test.reshape(-1, 224, 224, 3))
  
  # Compute cosine similarity between test samples and train samples
  cos_sim = cosine_similarity(X_test_features, X_train_features)
  predicted_labels_cos_sim = y_train[np.argmax(cos_sim, axis=1)]
  
  # Evaluate the cosine similarity classification
  print("Cosine Similarity Classification Report")
  print(classification_report(y_test, predicted_labels_cos_sim))
  
  # Function to preprocess the given image path
  def preprocess_image(image_path):
      print(f"Preprocessing image: {image_path}")
      img = load_img(image_path, target_size=(224, 224))
      img_array = img_to_array(img) / 255.0
      img_array = np.expand_dims(img_array, axis=0)
      return img_array
  
  # Function to extract features of the given image
  def extract_image_features(model, img_array):
      print("Extracting image features")
      features = model.predict(img_array)
      return features.reshape(1, -1)
  
  # Function to classify the given image
  def classify_image(image_path, model, X_train_features, y_train):
      img_array = preprocess_image(image_path)
      img_features = extract_image_features(model, img_array)
      cos_sim = cosine_similarity(img_features, X_train_features)
      predicted_label = y_train[np.argmax(cos_sim, axis=1)]
      return predicted_label[0]
  
  # Provide the path to the image you want to classify
  image_path = '/kaggle/input/testimage1/cast_def_0_138.jpeg'
  
  # Classify the given image
  predicted_quality = classify_image(image_path, model, X_train_features, y_train)
  print(f"The predicted quality for the image is: {'High' if predicted_quality == 1 else 'Low'}")
  
  # Display the given image
  img = load_img(image_path)
  plt.imshow(img)
  plt.axis('off')
  plt.show()
  
  # Set up Google Application Credentials for Dialogflow
  json_key_path = '/kaggle/input/keysfile/peschatbot45-obnl-7909ed2abbff.json'
  os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_key_path
  
  # Verify that the environment variable is set correctly
  print(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
  
  # Initialize Dialogflow client
  client = dialogflow.IntentsClient()
  project_id = 'peschatbot45-obnl'
  parent = f"projects/{project_id}/agent"
  print(parent)
  
  # Function to get existing intents
  def get_existing_intents():
      intents = client.list_intents(request={"parent": parent})
      return {intent.display_name: intent for intent in intents}
  
  # Get and print existing intents
  existing_intents = get_existing_intents()
  print("Existing Intents:", list(existing_intents.keys()))
  
  # FAQs data
  faqs = [
      {
          "question": "What are your opening hours?",
          "answers": [
              "We are open from 9 AM to 5 PM, Monday to Friday.",
              "Our business hours are from 9 AM to 5 PM, Monday to Friday."
          ]
      },
      {
          "question": "Where are you located?",
          "answers": [
              "We are located at 20 Ingram Street, Gotham opposite the Daily Planet.",
              "You will find us opposite the Daily Planet at 120 Ingram Street, Gotham."
          ]
      },
      {
          "question": "How can I contact customer service?",
          "answers": [
              "You can contact customer service at (123) 456-7890 or email us at support@example.com.",
              "For customer service call (123) 456-7890",
              "Please email us at support@example.com."
          ]
      },
      {
          "question": "What is your return policy?",
          "answers": [
              "Our return policy allows returns within 30 days of purchase with a receipt.",
              "If you have a receipt, you can return the items within 30 days of purchase as long they have not been used or damaged."
          ]
      }
  ]
  
  # Create or update intents
  for faq in faqs:
      create_or_update_intent(faq["question"], [faq["question"]], faq["answers"], existing_intents)
  
  # Detect intents for sample queries
  def detect_intent_texts(project_id, session_id, texts, language_code):
      session_client = dialogflow.SessionsClient()
      session = session_client.session_path(project_id, session_id)
      
      for text in texts:
          text_input = dialogflow.types.TextInput(text=text, language_code=language_code)
          query_input = dialogflow.types.QueryInput(text=text_input)
          
          response = session_client.detect_intent(session=session, query_input=query_input)
          print(f"Query text: {response.query_result.query_text}")
          print(f"Detected intent: {response.query_result.intent.display_name}")
          print(f"Response text: {response.query_result.fulfillment_text}")
          print("----------------------------------------------------------------")
  
  # Test queries
  test_queries = [
      "Hi", # This should trigger Welcome intent
      "What are your opening hours?",
      "Where are you located?",
      "How can I contact customer service?",
      "What is your return policy?",
      "What is your email address?",  # This should trigger the fallback intent
      "Do you offer discounts?"       # This should also trigger the fallback intent
  ]
  
  detect_intent_texts(project_id, "unique_session_id", test_queries, "en")
  
  # Add training phrases to specific intent
  intent_name = "What are your opening hours?"
  additional_training_phrases = [
      "When do you open?",
      "What time do you start business?",
      "Tell me your business hours.",
  ]
  
  # Add new training phrases
  create_or_update_intent(intent_name, additional_training_phrases, [], existing_intents)
  
  # Test the updated intent with a query
  test_query = [
      "What are your opening hours?",
      "When do you open?",
      "Tell me your business hours."
  ]
  
  detect_intent_texts(project_id, "unique_session_id", test_query, "en")
  
  # Delete all intents function
  def delete_all_intents():
      intents = client.list_intents(request={"parent": parent})
      for intent in intents:
          client.delete_intent(request={"name": intent.name})
      print("Deleted all intents.")
  
  # Delete all intents
  delete_all_intents()
  
  # Verify deletion
  existing_intents = get_existing_intents()
  print("Existing Intents After Deletion:", existing_intents)
      `,
      language: "python",
    },
    Step1: {
      code: `# Install necessary library
  !pip install google-cloud-dialogflow
  
  # Import libraries
  from google.cloud import dialogflow_v2 as dialogflow
  import os
  import pandas as pd
  import json
  
  print("done")
      `,
      language: "python",
    },
    Step2: {
      code: `json_key_path = '/kaggle/input/keysfile/peschatbot45-obnl-7909ed2abbff.json'
  os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_key_path
  
  # Verify that the environment variable is set correctly
  print(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
      `,
      language: "python",
    },
    Step3: {
      code: `client = dialogflow.IntentsClient()
  project_id = 'peschatbot45-obnl'
  parent = f"projects/{project_id}/agent"
  print(parent)
      `,
      language: "python",
    },
    Step4: {
      code: `def get_existing_intents():
      intents = client.list_intents(request={"parent": parent})
      return {intent.display_name: intent for intent in intents}
  
  # Get and print existing intents
  existing_intents = get_existing_intents()
  print("Existing Intents:", list(existing_intents.keys()))
      `,
      language: "python",
    },
    Step5: {
      code: `# FAQs data
  faqs = [
      {
          "question": "What are your opening hours?",
          "answers": [
              "We are open from 9 AM to 5 PM, Monday to Friday.",
              "Our business hours are from 9 AM to 5 PM, Monday to Friday."
          ]
      },
      {
          "question": "Where are you located?",
          "answers": [
              "We are located at 20 Ingram Street, Gotham opposite the Daily Planet.",
              "You will find us opposite the Daily Planet at 120 Ingram Street, Gotham."
          ]
      },
      {
          "question": "How can I contact customer service?",
          "answers": [
              "You can contact customer service at (123) 456-7890 or email us at support@example.com.",
              "For customer service call (123) 456-7890",
              "Please email us at support@example.com."
          ]
      },
      {
          "question": "What is your return policy?",
          "answers": [
              "Our return policy allows returns within 30 days of purchase with a receipt.",
              "If you have a receipt, you can return the items within 30 days of purchase as long they have not been used or damaged."
          ]
      }
  ]
  
  # Get existing intents
  existing_intents = get_existing_intents()
  
  # Create or update intents
  for faq in faqs:
      create_or_update_intent(faq["question"], [faq["question"]], faq["answers"], existing_intents)
      `,
      language: "python",
    },
    Step6: {
      code: `def detect_intent_texts(project_id, session_id, texts, language_code):
      session_client = dialogflow.SessionsClient()
      session = session_client.session_path(project_id, session_id)
      
      for text in texts:
          text_input = dialogflow.types.TextInput(text=text, language_code=language_code)
          query_input = dialogflow.types.QueryInput(text=text_input)
          
          response = session_client.detect_intent(session=session, query_input=query_input)
          print(f"Query text: {response.query_result.query_text}")
          print(f"Detected intent: {response.query_result.intent.display_name}")
          print(f"Response text: {response.query_result.fulfillment_text}")
          print("----------------------------------------------------------------")
      `,
      language: "python",
    },
    Step7: {
      code: `# Test queries
  test_queries = [
      "Hi", # This should trigger Welcome intent
      "What are your opening hours?",
      "Where are you located?",
      "How can I contact customer service?",
      "What is your return policy?",
      "What is your email address?",  # This should trigger the fallback intent
      "Do you offer discounts?"       # This should also trigger the fallback intent
  ]
  
  detect_intent_texts(project_id, "unique_session_id", test_queries, "en")
      `,
      language: "python",
    },
    Step8: {
      code: `# Example: Add training phrases to "What are your opening hours?" intent
  
  intent_name = "What are your opening hours?"
  additional_training_phrases = [
      "When do you open?",
      "What time do you start business?",
      "Tell me your business hours.",
  ]
  
  # Get existing intents
  existing_intents = get_existing_intents()
  
  # Add new training phrases
  create_or_update_intent(intent_name, additional_training_phrases, [], existing_intents)
  
  # Test the updated intent with a query
  test_query = [
      "What are your opening hours?",
      "When do you open?",
      "Tell me your business hours."
  ]
  
  detect_intent_texts(project_id, "unique_session_id", test_query, "en")
      `,
      language: "python",
    },
    Step9: {
      code: `def delete_all_intents():
      intents = client.list_intents(request={"parent": parent})
      for intent in intents:
          client.delete_intent(request={"name": intent.name})
      print("Deleted all intents.")
  
  # Delete all intents
  delete_all_intents()
  
  # Verify deletion
  existing_intents = get_existing_intents()
  print("Existing Intents After Deletion:", existing_intents)
      `,
      language: "python",
    },
  };
  
  export default codeSections;
