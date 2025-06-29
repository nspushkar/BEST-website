const codeSections = {
    full: {
      code: `import numpy as np
  import os
  import librosa
  import matplotlib.pyplot as plt
  from sklearn.preprocessing import StandardScaler
  from sklearn.cluster import DBSCAN
  from sklearn.decomposition import PCA
  from sklearn.neighbors import NearestNeighbors
  
  # Paths to the dataset
  TRAIN_PATH = '/kaggle/input/dataset/dev_gearbox/gearbox/train'
  TEST_PATH = '/kaggle/input/dataset/dev_gearbox/gearbox/test'
  
  def load_audio_files(path):
      audio_files = []
      for root, _, files in os.walk(path):
          for file in files:
              if file.endswith('.wav'):
                  file_path = os.path.join(root, file)
                  audio_files.append(file_path)
      return audio_files
  
  def extract_features(file_list, n_fft=1024, hop_length=512):
      features = []
      for file_path in file_list:
          y, sr = librosa.load(file_path, sr=None)
          spectrogram = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=False)
          magnitude, _ = librosa.magphase(spectrogram)
          magnitude_db = librosa.amplitude_to_db(magnitude, ref=1e-6)
          features.append(magnitude_db.flatten())
      return np.array(features)
  
  # Load audio files
  train_files = load_audio_files(TRAIN_PATH)
  test_files = load_audio_files(TEST_PATH)
  
  # Extract features
  X_train = extract_features(train_files)
  X_test = extract_features(test_files)
  
  # Standardize the data
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)
  
  # Reduce the dimensionality of the data using PCA for visualization purposes
  pca = PCA(n_components=2)
  X_train_pca = pca.fit_transform(X_train_scaled)
  X_test_pca = pca.transform(X_test_scaled)
  
  # Visualize the data distribution using PCA components
  plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], s=5)
  plt.xlabel('PCA Component 1')
  plt.ylabel('PCA Component 2')
  plt.title('PCA of Training Data')
  plt.show()
  
  # Select an eps value from the plot and apply DBSCAN
  eps_value = 480  # Chosen based on the k-distance plot elbow
  dbscan = DBSCAN(eps=eps_value, min_samples=20)
  dbscan.fit(X_train_scaled)
  
  # Predict the cluster for each sample in the test set
  test_clusters = dbscan.fit_predict(X_test_scaled)
  
  print(f"Cluster assignments for the test set: {test_clusters}")
  
  # Visualize the clustering using PCA components
  plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=test_clusters, cmap='viridis', s=5)
  plt.xlabel('PCA Component 1')
  plt.ylabel('PCA Component 2')
  plt.title(f'DBSCAN Clustering with eps={eps_value}')
  plt.show()
  `,
      language: 'python',
    },
  };
  
  export default codeSections;
  