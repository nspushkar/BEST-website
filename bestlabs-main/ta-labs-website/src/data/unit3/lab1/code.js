const codeSections = {
    Step1: {
      code: `# Install libraries if not already installed
  !pip install tensorflow scikit-learn`,
      language: 'python',
    },
    ImportLibs: {
      code: `import numpy as np
  import tensorflow as tf
  from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
  from tensorflow.keras.applications import VGG16
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import StandardScaler
  from sklearn.metrics.pairwise import cosine_similarity
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.cluster import KMeans
  from sklearn.metrics import classification_report
  import matplotlib.pyplot as plt`,
      language: 'python',
    },
    DefinePath: {
      code: `dataset_path = '/kaggle/input/real-life-industrial-dataset-of-casting-product'`,
      language: 'python',
    },
    LoadDataset: {
      code: `datagen = ImageDataGenerator(rescale=1./255)
  dataset = datagen.flow_from_directory(dataset_path, target_size=(224, 224), batch_size=32, class_mode='binary')`,
      language: 'python',
    },
    LoadPrepData: {
      code: `X, y = load_data(dataset)`,
      language: 'python',
    },
    SplitData: {
      code: `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)`,
      language: 'python',
    },
    NormaliseReshape: {
      code: `scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train.reshape(-1, 224 * 224 * 3))
  X_test = scaler.transform(X_test.reshape(-1, 224 * 224 * 3))`,
      language: 'python',
    },
    LoadVGG16: {
      code: `base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
  model = tf.keras.Model(inputs=base_model.input, outputs=base_model.getLayer('block5_pool').output)`,
      language: 'python',
    },
    ExtractFeatures: {
      code: `X_train_features = extract_features(model, X_train.reshape(-1, 224, 224, 3))
  X_test_features = extract_features(model, X_test.reshape(-1, 224, 224, 3))`,
      language: 'python',
    },
    ComputeCosSim: {
      code: `cos_sim = cosine_similarity(X_test_features, X_train_features)
  predicted_labels_cos_sim = y_train[np.argmax(cos_sim, axis=1)]`,
      language: 'python',
    },
    EvaluateClass: {
      code: `print("Cosine Similarity Classification Report")
  print(classification_report(y_test, predicted_labels_cos_sim))`,
      language: 'python',
    },
    PreprocessImage: {
      code: `def preprocess_image(image_path):
      print(f"Preprocessing image: {image_path}")
      img = load_img(image_path, target_size=(224, 224))
      img_array = img_to_array(img) / 255.0
      img_array = np.expand_dims(img_array, axis=0)
      return img_array
  
  def extract_image_features(model, img_array):
      print("Extracting image features")
      features = model.predict(img_array)
      return features.reshape(1, -1)`,
      language: 'python',
    },
    ClassifyImage: {
      code: `image_path = '/kaggle/input/testimage1/cast_def_0_138.jpeg'
  predicted_quality = classify_image(image_path, model, X_train_features, y_train)
  print(f"The predicted quality for the image is: {'High' if predicted_quality == 1 else 'Low'}")`,
      language: 'python',
    },
    full: {
      code: `# Install libraries if not already installed
  !pip install tensorflow scikit-learn
  
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
  plt.show()`,
      language: 'python',
    },
  };
  
  export default codeSections;
  