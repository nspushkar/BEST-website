const codeSections = {
    Step1: {
      code: `import pandas as pd
  import numpy as np
  from sklearn.preprocessing import StandardScaler
  from sklearn.cluster import KMeans
  from sklearn.decomposition import PCA
  import matplotlib.pyplot as plt
  import seaborn as sns
  
  # loading the data from csv file to a Pandas DataFrame
  customer_data = pd.read_csv('Mall_Customers.csv')
  
  # first 5 rows in the dataframe
  customer_data.head()
  
  # finding the number of rows and columns
  customer_data.shape
  
  # getting some informations about the dataset
  customer_data.info()
  
  # checking for missing values
  customer_data.isnull().sum()
  
  relevant_columns = ['Annual Income (k$)', 'Spending Score (1-100)']
  X = customer_data.iloc[:,[3,4]].values
  
  print(X)
  
  # Normalizing the data
  scaler = StandardScaler()
  scaled_data = scaler.fit_transform(X)`,
      language: 'python'
    },
    elbow: {
      code: `# finding wcss value for different number of clusters
  
  wcss = []
  
  for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
  
    wcss.append(kmeans.inertia_)
    
  # plot an elbow graph
  
  sns.set()
  plt.plot(range(1,11), wcss)
  plt.title('The Elbow Point Graph')
  plt.xlabel('Number of Clusters')
  plt.ylabel('WCSS')
  plt.show()`,
      language: 'python'
    },
    ClusterSummary: {
      code: `# Select only relevant columns for cluster analysis
  cluster_summary = customer_data.groupby('Cluster')[relevant_columns].mean()
  print(cluster_summary)`,
      language: 'python'
    },
    Gender: {
      code: `# Count of males and females in each cluster
  gender_count = customer_data.groupby(['Cluster', 'Genre']).size().unstack(fill_value=0)
  print(gender_count)`,
      language: 'python'
    },
    NoCustomer: {
      code: `# Visualizing the number of customers in each cluster
  plt.figure(figsize=(10, 6))
  sns.countplot(x='Cluster', data=customer_data)
  plt.title('Number of Customers in Each Cluster')
  plt.xlabel('Cluster')
  plt.ylabel('Number of Customers')
  plt.show()`,
      language: 'python'
    },
    OrgVisual: {
      code: `# Visualizing the clusters (Original Data)
  plt.figure(figsize=(16, 8))
  colors = ['green', 'red', 'yellow', 'violet', 'blue']
  
  # Original data visualization
  plt.subplot(1, 2, 1)
  for i in range(5):
      plt.scatter(X[Y == i, 0], X[Y == i, 1], s=50, c=colors[i], label=f'Cluster {i+1}')
  plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='cyan', label='Centroids')
  plt.title('Customer Groups (Original Data)')
  plt.xlabel('Annual Income')
  plt.ylabel('Spending Score')
  plt.legend()`,
      language: 'python'
    },
    PcaK: {
      code: `# PCA for dimensionality reduction
  pca = PCA(n_components=2)
  pca_components = pca.fit_transform(X)
  
  # Visualize the PCA-reduced clusters
  plt.scatter(pca_components[:, 0], pca_components[:, 1], c=Y, cmap='viridis')
  plt.xlabel('Principal Component 1')
  plt.ylabel('Principal Component 2')
  plt.title('Customer Segmentation using PCA')
  plt.legend()
  plt.show()`,
      language: 'python'
    },
    full: {
      code: `import pandas as pd
  import numpy as np
  from sklearn.preprocessing import StandardScaler
  from sklearn.cluster import KMeans
  from sklearn.decomposition import PCA
  import matplotlib.pyplot as plt
  import seaborn as sns
  
  # loading the data from csv file to a Pandas DataFrame
  customer_data = pd.read_csv('Mall_Customers.csv')
  
  # first 5 rows in the dataframe
  customer_data.head()
  
  # finding the number of rows and columns
  customer_data.shape
  
  # getting some informations about the dataset
  customer_data.info()
  
  # checking for missing values
  customer_data.isnull().sum()
  
  relevant_columns = ['Annual Income (k$)', 'Spending Score (1-100)']
  X = customer_data.iloc[:,[3,4]].values
  
  print(X)
  
  # Normalizing the data
  scaler = StandardScaler()
  scaled_data = scaler.fit_transform(X)
  
  # finding wcss value for different number of clusters
  
  wcss = []
  
  for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
  
    wcss.append(kmeans.inertia_)
    
  # plot an elbow graph
  
  sns.set()
  plt.plot(range(1,11), wcss)
  plt.title('The Elbow Point Graph')
  plt.xlabel('Number of Clusters')
  plt.ylabel('WCSS')
  plt.show()
  
  kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
  
  # return a label for each data point based on their cluster
  Y = kmeans.fit_predict(X)
  
  # Adding the cluster labels to the original DataFrame
  customer_data['Cluster'] = Y
  
  print(Y)
  
  # Select only relevant columns for cluster analysis
  cluster_summary = customer_data.groupby('Cluster')[relevant_columns].mean()
  print(cluster_summary)
  
  # Count of males and females in each cluster
  gender_count = customer_data.groupby(['Cluster', 'Genre']).size().unstack(fill_value=0)
  print(gender_count)
  
  # Visualizing the number of customers in each cluster
  plt.figure(figsize=(10, 6))
  sns.countplot(x='Cluster', data=customer_data)
  plt.title('Number of Customers in Each Cluster')
  plt.xlabel('Cluster')
  plt.ylabel('Number of Customers')
  plt.show()
  
  # Visualizing the clusters (Original Data)
  plt.figure(figsize=(16, 8))
  colors = ['green', 'red', 'yellow', 'violet', 'blue']
  
  # Original data visualization
  plt.subplot(1, 2, 1)
  for i in range(5):
      plt.scatter(X[Y == i, 0], X[Y == i, 1], s=50, c=colors[i], label=f'Cluster {i+1}')
  plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='cyan', label='Centroids')
  plt.title('Customer Groups (Original Data)')
  plt.xlabel('Annual Income')
  plt.ylabel('Spending Score')
  plt.legend()
  
  # PCA for dimensionality reduction
  pca = PCA(n_components=2)
  pca_components = pca.fit_transform(X)
  
  # Visualize the PCA-reduced clusters
  plt.scatter(pca_components[:, 0], pca_components[:, 1], c=Y, cmap='viridis')
  plt.xlabel('Principal Component 1')
  plt.ylabel('Principal Component 2')
  plt.title('Customer Segmentation using PCA')
  plt.legend()
  plt.show()`,
      language: 'python'
    },
    visualisation: {
      code: `# Visualizing the number of customers in each cluster
  plt.figure(figsize=(10, 6))
  sns.countplot(x='Cluster', data=customer_data)
  plt.title('Number of Customers in Each Cluster')
  plt.xlabel('Cluster')
  plt.ylabel('Number of Customers')
  plt.show()
  
  # Visualizing the clusters (Original Data)
  plt.figure(figsize=(16, 8))
  colors = ['green', 'red', 'yellow', 'violet', 'blue']
  
  # Original data visualization
  plt.subplot(1, 2, 1)
  for i in range(5):
      plt.scatter(X[Y == i, 0], X[Y == i, 1], s=50, c=colors[i], label=f'Cluster {i+1}')
  plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='cyan', label='Centroids')
  plt.title('Customer Groups (Original Data)')
  plt.xlabel('Annual Income')
  plt.ylabel('Spending Score')
  plt.legend()
  
  # PCA for dimensionality reduction
  pca = PCA(n_components=2)
  pca_components = pca.fit_transform(X)
  
  # Visualize the PCA-reduced clusters
  plt.scatter(pca_components[:, 0], pca_components[:, 1], c=Y, cmap='viridis')
  plt.xlabel('Principal Component 1')
  plt.ylabel('Principal Component 2')
  plt.title('Customer Segmentation using PCA')
  plt.legend()
  plt.show()`,
  language: 'python'
    },
    clustering: {
      code: `# finding wcss value for different number of clusters
  
  wcss = []
  
  for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
  
    wcss.append(kmeans.inertia_)
    
  # plot an elbow graph
  
  sns.set()
  plt.plot(range(1,11), wcss)
  plt.title('The Elbow Point Graph')
  plt.xlabel('Number of Clusters')
  plt.ylabel('WCSS')
  plt.show()
  
  kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
  
  # return a label for each data point based on their cluster
  Y = kmeans.fit_predict(X)
  
  # Adding the cluster labels to the original DataFrame
  customer_data['Cluster'] = Y
  
  print(Y)`,
  language: 'python'
    }
  };
  
  export default codeSections;
  