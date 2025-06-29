const codeSections = {
    step1: {
      code: `# Data Preparation
  
  train_img_path = '/kaggle/input/car-damage-severity-dataset/data3a/training'
  test_img_path = '/kaggle/input/car-damage-severity-dataset/data3a/validation'
  
  batch_size = 32
  img_height = 224
  img_width = 224
  
  train_data_gen = ImageDataGenerator(rescale=1 / 255.0,
          rotation_range=20,
          zoom_range=0.05,
          width_shift_range=0.05,
          height_shift_range=0.05,
          shear_range=0.05,
          horizontal_flip=True,
          validation_split=0.20,) 
  
  # Use flow_from_directory for the training dataset
  train_ds = train_data_gen.flow_from_directory(
      train_img_path,
      target_size=(img_height, img_width),
      batch_size=batch_size,
      class_mode='categorical',  
      subset='training', 
      seed=123,
      shuffle=True  
  )
  valid_ds = train_data_gen.flow_from_directory(
      train_img_path,
      target_size=(img_height, img_width),
      batch_size=batch_size,
      class_mode='categorical',  
      subset='validation', 
      seed=123,
      shuffle=True  
  )
  
  test_data_gen = ImageDataGenerator(rescale=1./255,)  # You may adjust other parameters as needed
  
  # Use flow_from_directory for the test dataset
  test_ds = test_data_gen.flow_from_directory(
      test_img_path,
      target_size=(img_height, img_width),
      batch_size=batch_size,
      class_mode='categorical',
      shuffle=False
  )
  
  cl=test_ds.class_indices
  print(cl)
  
  def plot_images_from_dataset(dataset, num_images=9):
      # Fetch a batch of images and labels from the dataset
      images, labels = next(iter(dataset))
  
      plt.figure(figsize=(10, 10))
      for i in range(min(num_images, len(images))):  
          plt.subplot(3, 3, i + 1)
          plt.imshow(images[i])
          
          # Map the label index back to the original class name
          label_index = labels[i].argmax()  # Assumes one-hot encoding
          class_name = next(key for key, value in cl.items() if value == label_index)
          
          plt.title(f"Class: {class_name}")
          plt.axis("off")
      plt.show()
  
  # Assuming test_ds is your dataset
  plot_images_from_dataset(test_ds)
      `,
      language: 'python',
    },
    step2: {
      code: `# CNN USING PRETRAINED EFF NET
  
  img_size = (224, 224)
  lr = 0.001
  class_count = 3
  
  img_shape = (img_size[0], img_size[1], 3)
  
  base_model = DenseNet169(include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')
  base_model.trainable = True
  x = base_model.output
  x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
  x = Dense(256, kernel_regularizer=regularizers.l2(l=0.016), 
            activity_regularizer=regularizers.l1(0.006),
            bias_regularizer=regularizers.l1(0.006), activation='relu')(x)
  x = Dropout(rate=.4, seed=123)(x)
  output = Dense(class_count, activation='softmax')(x)
  model_eff = Model(inputs=base_model.input, outputs=output)
  model_eff.compile(Adamax(learning_rate=lr), loss='categorical_crossentropy', 
                metrics=['accuracy','AUC'])
  
  epochs=50
  
  # Train the model
  history_eff = model_eff.fit(
      train_ds,
      epochs=epochs,
      validation_data=valid_ds,  
      verbose=1,
  )
  
  # Save training and validation histories for later analysis
  all_train_histories = [history_eff.history['accuracy']]
  all_val_histories = [history_eff.history['val_accuracy']]
  
  model_eff.save('model_eff.h5')
  
  from tensorflow.keras.preprocessing.image import load_img, img_to_array
  import numpy as np
  
  # Function to preprocess image
  def preprocess_image(image_path, target_size=(224, 224)):
      img = load_img(image_path, target_size=target_size)
      img_array = img_to_array(img)
      img_array = img_array / 255.0  # Normalize pixel values
      img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
      return img_array
  
  # Function to predict and display the image with its predicted class
  def predict_and_display_image(image_path, model):
      # Preprocess the image
      processed_image = preprocess_image(image_path)
      
      # Make predictions
      predictions = model.predict(processed_image)
      
      # Get the predicted class label
      predicted_class_index = np.argmax(predictions)
      class_name = next(key for key, value in cl.items() if value == predicted_class_index)
      
      # Display the image with its predicted class
      img = load_img(image_path)
      plt.imshow(img)
      plt.title(f"Predicted Class: {class_name}")
      plt.axis("off")
      plt.show()
  
  # Path to your own image
  your_image_path = '/kaggle/input/test-2-minor/0015.JPEG'  # Change this to your image path
  
  # Predict and display the image
  predict_and_display_image(your_image_path, model_eff)
      `,
      language: 'python',
    },
    step3: {
      code: `# Define the upload button and output widget
  upload_btn = widgets.FileUpload(accept='image/*', multiple=False)
  output = widgets.Output()
  
  # Function to handle the image upload and classification
  def on_upload_change(change):
      for filename, file_info in change['new'].items():
          # Save the uploaded image to a temporary path
          with open('uploaded_image.jpg', 'wb') as f:
              f.write(file_info['content'])
          
          # Display the image and prediction
          with output:
              output.clear_output()  # Clear previous outputs
              predict_and_display_image('uploaded_image.jpg', model_eff)
  
  # Attach the function to the upload button
  upload_btn.observe(on_upload_change, names='value')
  
  # Display the upload button and output widget
  display(upload_btn, output)
  
  plt.plot(history_eff.history['accuracy'], label='Training Accuracy')
  plt.plot(history_eff.history['val_accuracy'], label='Validation Accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.title('Training and Validation Accuracy over Epochs')
  plt.show()
  
  # Testing code
  test_accuracy = model_eff.evaluate(test_ds)
  
  # Confusion matrix
  true_labels = test_ds.classes
  predictions = model_eff.predict(test_ds)
  predicted_labels = np.argmax(predictions, axis=1)
  
  sns.heatmap(confusion_matrix(true_labels, predicted_labels), annot=True)
  
  # Print classification report
  print(classification_report(true_labels, predicted_labels))
  
  plt.title('Confusion Matrix')
  plt.xlabel('Predicted')
  plt.ylabel('Actual Classes')
  plt.show()
      `,
      language: 'python',
    },
    full: {
      code: `# Import Libraries
  
  import numpy as np
  import pandas as pd
  import cv2
  import tensorflow as tf
  import matplotlib.pyplot as plt
  from tensorflow.keras.preprocessing.image import ImageDataGenerator
  from tensorflow.keras.callbacks import EarlyStopping
  from tensorflow.keras.applications import ResNet50
  from tensorflow.keras import layers, models, optimizers
  from sklearn.model_selection import KFold
  from tensorflow.keras import regularizers
  from tensorflow.keras.applications import DenseNet169
  from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
  from tensorflow.keras.models import Model
  from tensorflow.keras.optimizers import Adamax
  from tensorflow_addons.metrics import F1Score
  import ipywidgets as widgets
  from IPython.display import display
  from tensorflow.keras.preprocessing.image import load_img, img_to_array
  import seaborn as sns
  from sklearn.metrics import confusion_matrix
  from sklearn.metrics import classification_report
  
  # Data Preparation
  
  train_img_path = '/kaggle/input/car-damage-severity-dataset/data3a/training'
  test_img_path = '/kaggle/input/car-damage-severity-dataset/data3a/validation'
  
  batch_size = 32
  img_height = 224
  img_width = 224
  
  train_data_gen = ImageDataGenerator(rescale=1 / 255.0,
          rotation_range=20,
          zoom_range=0.05,
          width_shift_range=0.05,
          height_shift_range=0.05,
          shear_range=0.05,
          horizontal_flip=True,
          validation_split=0.20,) 
  
  train_ds = train_data_gen.flow_from_directory(
      train_img_path,
      target_size=(img_height, img_width),
      batch_size=batch_size,
      class_mode='categorical',  
      subset='training', 
      seed=123,
      shuffle=True  
  )
  valid_ds = train_data_gen.flow_from_directory(
      train_img_path,
      target_size=(img_height, img_width),
      batch_size=batch_size,
      class_mode='categorical',  
      subset='validation', 
      seed=123,
      shuffle=True  
  )
  
  test_data_gen = ImageDataGenerator(rescale=1./255,) 
  
  test_ds = test_data_gen.flow_from_directory(
      test_img_path,
      target_size=(img_height, img_width),
      batch_size=batch_size,
      class_mode='categorical',
      shuffle=False
  )
  
  cl=test_ds.class_indices
  print(cl)
  
  def plot_images_from_dataset(dataset, num_images=9):
      images, labels = next(iter(dataset))
      plt.figure(figsize=(10, 10))
      for i in range(min(num_images, len(images))):  
          plt.subplot(3, 3, i + 1)
          plt.imshow(images[i])
          label_index = labels[i].argmax()
          class_name = next(key for key, value in cl.items() if value == label_index)
          plt.title(f"Class: {class_name}")
          plt.axis("off")
      plt.show()
  
  plot_images_from_dataset(test_ds)
  
  # CNN USING PRETRAINED EFF NET
  
  img_size = (224, 224)
  lr = 0.001
  class_count = 3
  
  img_shape = (img_size[0], img_size[1], 3)
  
  base_model = DenseNet169(include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')
  base_model.trainable = True
  x = base_model.output
  x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
  x = Dense(256, kernel_regularizer=regularizers.l2(l=0.016), 
            activity_regularizer=regularizers.l1(0.006),
            bias_regularizer=regularizers.l1(0.006), activation='relu')(x)
  x = Dropout(rate=.4, seed=123)(x)
  output = Dense(class_count, activation='softmax')(x)
  model_eff = Model(inputs=base_model.input, outputs=output)
  model_eff.compile(Adamax(learning_rate=lr), loss='categorical_crossentropy', 
                metrics=['accuracy','AUC'])
  
  epochs=50
  
  history_eff = model_eff.fit(
      train_ds,
      epochs=epochs,
      validation_data=valid_ds,  
      verbose=1,
  )
  
  all_train_histories = [history_eff.history['accuracy']]
  all_val_histories = [history_eff.history['val_accuracy']]
  
  model_eff.save('model_eff.h5')
  
  # Function to preprocess image
  def preprocess_image(image_path, target_size=(224, 224)):
      img = load_img(image_path, target_size=target_size)
      img_array = img_to_array(img)
      img_array = img_array / 255.0
      img_array = np.expand_dims(img_array, axis=0)
      return img_array
  
  # Function to predict and display the image with its predicted class
  def predict_and_display_image(image_path, model):
      processed_image = preprocess_image(image_path)
      predictions = model.predict(processed_image)
      predicted_class_index = np.argmax(predictions)
      class_name = next(key for key, value in cl.items() if value == predicted_class_index)
      img = load_img(image_path)
      plt.imshow(img)
      plt.title(f"Predicted Class: {class_name}")
      plt.axis("off")
      plt.show()
  
  your_image_path = '/kaggle/input/test-2-minor/0015.JPEG'  # Change this to your image path
  predict_and_display_image(your_image_path, model_eff)
  
  # Define the upload button and output widget
  upload_btn = widgets.FileUpload(accept='image/*', multiple=False)
  output = widgets.Output()
  
  def on_upload_change(change):
      for filename, file_info in change['new'].items():
          with open('uploaded_image.jpg', 'wb') as f:
              f.write(file_info['content'])
          with output:
              output.clear_output()
              predict_and_display_image('uploaded_image.jpg', model_eff)
  
  upload_btn.observe(on_upload_change, names='value')
  display(upload_btn, output)
  
  plt.plot(history_eff.history['accuracy'], label='Training Accuracy')
  plt.plot(history_eff.history['val_accuracy'], label='Validation Accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.title('Training and Validation Accuracy over Epochs')
  plt.show()
  
  test_accuracy = model_eff.evaluate(test_ds)
  
  true_labels = test_ds.classes
  predictions = model_eff.predict(test_ds)
  predicted_labels = np.argmax(predictions, axis=1)
  
  sns.heatmap(confusion_matrix(true_labels, predicted_labels), annot=True)
  
  print(classification_report(true_labels, predicted_labels))
  
  plt.title('Confusion Matrix')
  plt.xlabel('Predicted')
  plt.ylabel('Actual Classes')
  plt.show()
      `,
      language: 'python',
    },
  };
  
  export default codeSections;
  