const codeSections = {
    UnderstandingDecisionTrees: {
      code: `# Initialize the DecisionTreeClassifier
  clf = DecisionTreeClassifier(random_state = 42, ccp_alpha = ccp_alpha)
  clf.fit(X_train,y_train)
  `,
      language: 'python',
    },
    GeneratingDataset: {
      code: `np.random.seed(42) # For reproducibility
  num_samples = 1000
  # Generate random values for depth, rate, and precision using numpy's uniform distribution
  depth = np.random.uniform(1, 100, num_samples)
  rate = np.random.uniform(1, 1000, num_samples)
  precision = np.random.uniform(0, 1, num_samples)
  # Create a DataFrame with the generated data
  data = pd.DataFrame({'depth': depth, 'precision': precision, 'rate': rate})
  # Print the first few rows of the generated dataset
  print("Generated Dataset:")
  print(data.head())
  
  # Classification function to assign classes based on depth, precision, and rate
  def classify(depth, precision, rate):
    if depth > 80 and precision > 0.80:
      return "very good"
    elif depth > 60 and precision > 0.60 and rate > 600:
      return "good"
    elif depth > 40 and precision > 0.40 and rate > 400:
      return "ok"
    elif depth > 20 and precision > 0.20 and rate > 200:
      return "bad"
    else:
      return "very bad"
  # Assign 'classes' based on classification function
  data['classes'] = data.apply(lambda row: classify(row['depth'], row['precision'], row['rate']), axis=1)
  # Print dataset before splitting
  print("Dataset before splitting:")
  print(data.head())
  # Split Dataset into Features (X) and Target (y)
  X = data[['depth', 'precision', 'rate']]
  y = data['classes']
  # Split into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
  `,
      language: 'python',
    },
    BuildingDecisionTree: {
      code: `train_scores = [clf.score(X_train,y_train) for clf in clfs]
  test_scores = [clf.score(X_test, y_test) for clf in clfs]
  plt.figure(figsize=(10, 6))
  plt.plot(ccp_alphas, train_scores, marker='o', label='train', drawstyle="steps-post")
  plt.plot(ccp_alphas, test_scores, marker='o', label='test', drawstyle="steps-post")
  plt.xlabel('alpha')
  plt.ylabel('accuracy')
  plt.title('Accuracy vs alpha for training and testing sets')
  plt.legend()
  plt.show()
  
  optimal_clf = clfs[np.argmax(test_scores)]
  # Display the tree structure
  plt.figure(figsize=(20, 10))
  plot_tree(optimal_clf, filled=True, feature_names=['depth', 'precision', 'rate'], class_names=[str(i) for i in range(1, 6)])
  plt.show()
  `,
      language: 'python',
    },
    UnderstandingPruning: {
      code: `# Pruning in Decision Trees
  # Example: Cost Complexity Pruning (ccp)
  path = clf.cost_complexity_pruning_path(X_train,y_train)
  ccp_alphas = path.ccp_alphas
  impurities = path.impurities
  
  clfs = []
  for ccp_alpha in ccp_alphas:
      clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
      clf.fit(X_train, y_train)
      clfs.append(clf)
  
  train_scores = [clf.score(X_train, y_train) for clf in clfs]
  test_scores = [clf.score(X_test, y_test) for clf in clfs]
  
  plt.figure(figsize=(10, 6))
  plt.plot(ccp_alphas, train_scores, marker='o', label='train', drawstyle="steps-post")
  plt.plot(ccp_alphas, test_scores, marker='o', label='test', drawstyle="steps-post")
  plt.xlabel('alpha')
  plt.ylabel('accuracy')
  plt.title('Accuracy vs alpha for training and testing sets')
  plt.legend()
  plt.show()
  
  # Find optimal model based on test set performance
  optimal_clf = clfs[np.argmax(test_scores)]
  # Display the tree structure
  plt.figure(figsize=(20, 10))
  plot_tree(optimal_clf, filled=True, feature_names=['depth', 'precision', 'rate'], class_names=[str(i) for i in range(1, 6)])
  plt.show()
  `,
      language: 'python',
    },
    ReadingEvaluatingTree: {
      code: `# Reading and Evaluating the Tree
  
  # Generate test data
  num_test_samples = 100
  test_depth = np.random.uniform(1, 100, num_test_samples)
  test_precision = np.random.uniform(0, 1, num_test_samples)
  test_speed = np.random.uniform(10, 1000, num_test_samples)
  
  test_data = pd.DataFrame({'depth': test_depth, 'precision': test_precision, 'rate': test_speed})
  
  # Classify test data
  test_data['actual_level'] = test_data.apply(lambda row: classify(row['depth'], row['precision'], row['rate']), axis=1)
  
  # Predict using the trained decision tree model
  test_X = test_data[['depth', 'precision', 'rate']]
  test_data['predicted_level'] = optimal_clf.predict(test_X)
  
  # Display test data with actual and predicted levels
  print(test_data.head())
  
  # Evaluate the model on test data
  accuracy = accuracy_score(test_data['actual_level'], test_data['predicted_level'])
  conf_matrix = confusion_matrix(test_data['actual_level'], test_data['predicted_level'])
  
  print(f"Accuracy on test data: {accuracy * 100:.2f}%")
  print("Confusion Matrix:")
  print(conf_matrix)
  `,
      language: 'python',
    },
    full: {
      code: `import numpy as np
  import pandas as pd
  from sklearn.model_selection import train_test_split, cross_val_score
  from sklearn.tree import DecisionTreeClassifier, plot_tree
  from sklearn.tree import export_text
  from sklearn.metrics import classification_report
  import matplotlib.pyplot as plt
  from sklearn.metrics import accuracy_score, confusion_matrix
  from IPython.display import Image, display
  from sklearn.model_selection import GridSearchCV
  
  np.random.seed(42)  # For reproducibility
  num_samples = 1000
  
  # Generate random values for depth, rate, and precision using numpy's uniform distribution
  depth = np.random.uniform(1, 100, num_samples)
  rate = np.random.uniform(1, 1000, num_samples)
  precision = np.random.uniform(0, 1, num_samples)
  
  # Create a DataFrame with the generated data
  data = pd.DataFrame({'depth': depth, 'precision': precision, 'rate': rate})
  
  # Print the first few rows of the generated dataset
  print("Generated Dataset:")
  print(data.head())
  
  # Classification function to assign classes based on depth, precision, and rate
  def classify(depth, precision, rate):
      if depth > 80 and precision > 0.80:
          return "very good"
      elif depth > 60 and precision > 0.60 and rate > 600:
          return "good"
      elif depth > 40 and precision > 0.40 and rate > 400:
          return "ok"
      elif depth > 20 and precision > 0.20 and rate > 200:
          return "bad"
      else:
          return "very bad"
  
  # Assign 'classes' based on classification function
  data['classes'] = data.apply(lambda row: classify(row['depth'], row['precision'], row['rate']), axis=1)
  
  # Print dataset before splitting
  print("Dataset before splitting:")
  print(data.head())
  
  # Step 2: Split Dataset into Features (X) and Target (y)
  X = data[['depth', 'precision', 'rate']]
  y = data['classes']
  
  # Step 3: Split into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
  
  # Implement Pruning (Cost Complexity Pruning)
  path = clf.cost_complexity_pruning_path(X_train,y_train)
  ccp_alphas = path.ccp_alphas
  impurities = path.impurities
  
  clfs = []
  for ccp_alpha in ccp_alphas:
      clf = DecisionTreeClassifier(random_state = 42, ccp_alpha = ccp_alpha)
      clf.fit(X_train,y_train)
      clfs.append(clf)
  
  train_scores = [clf.score(X_train,y_train) for clf in clfs]
  test_scores  = [clf.score(X_test, y_test) for clf in clfs]
  
  plt.figure(figsize=(10, 6))
  plt.plot(ccp_alphas, train_scores, marker='o', label='train', drawstyle="steps-post")
  plt.plot(ccp_alphas, test_scores, marker='o', label='test', drawstyle="steps-post")
  plt.xlabel('alpha')
  plt.ylabel('accuracy')
  plt.title('Accuracy vs alpha for training and testing sets')
  plt.legend()
  plt.show()
  
  optimal_clf = clfs[np.argmax(test_scores)]
  
  # Display the tree structure
  plt.figure(figsize=(20, 10))
  plot_tree(optimal_clf, filled=True, feature_names=['depth', 'precision', 'rate'], class_names=[str(i) for i in range(1, 6)])
  plt.show()
  
  # Print the decision rules
  tree_rules = export_text(optimal_clf, feature_names=['depth', 'precision', 'rate'])
  print(tree_rules)
  
  # Evaluate the optimal model
  y_pred = optimal_clf.predict(X_test)
  print(classification_report(y_test, y_pred))
  
  # Generate test data
  num_test_samples = 100
  test_depth = np.random.uniform(1, 100, num_test_samples)
  test_precision = np.random.uniform(0, 1, num_test_samples)
  test_speed = np.random.uniform(10, 1000, num_test_samples)
  
  test_data = pd.DataFrame({'depth': test_depth, 'precision': test_precision, 'rate': test_speed})
  
  # Classify test data
  test_data['actual_level'] = test_data.apply(lambda row: classify(row['depth'], row['precision'], row['rate']), axis=1)
  
  # Predict using the trained decision tree model
  test_X = test_data[['depth', 'precision', 'rate']]
  test_data['predicted_level'] = optimal_clf.predict(test_X)
  
  # Display test data with actual and predicted levels
  print(test_data.head())
  
  
  # Evaluate the model on test data
  accuracy = accuracy_score(test_data['actual_level'], test_data['predicted_level'])
  conf_matrix = confusion_matrix(test_data['actual_level'], test_data['predicted_level'])
  
  print(f"Accuracy on test data: {accuracy * 100:.2f}%")
  print("Confusion Matrix:")
  print(conf_matrix)
  
  test_X = pd.DataFrame({'depth': [80],
                         'precision': [0.9],
                         'rate': [860.228603]})
  
  # Predict using the trained decision tree model
  predicted_level = optimal_clf.predict(test_X)
  
  # Print the predicted level
  print(f"Predicted Level: {predicted_level[0]}")
  
  # 1. single hyper-parameter pre pruned tree
  #Let us implement single hyper-parameter max_depth.
  mean_scores = []
  depth_range = range(1, 21)
  
  for depth in depth_range:
      clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
      scores = cross_val_score(clf, X_train, y_train, cv=5)
      mean_scores.append(scores.mean())
      
  plt.figure(figsize=(10, 6))
  plt.plot(depth_range, mean_scores, marker='o')
  plt.xlabel('max_depth')
  plt.ylabel('Cross-Validated Accuracy')
  plt.title('Accuracy vs max_depth')
  plt.show()
  
  # Find the optimal max_depth
  optimal_depth = depth_range[np.argmax(mean_scores)]
  print(f'Optimal max_depth: {optimal_depth}')
  
  # Train the decision tree with the optimal max_depth
  spre_tree=DecisionTreeClassifier(max_depth=optimal_depth, random_state=42)
  spre_tree.fit(X_train, y_train)
  
  # Display the tree structure
  plt.figure(figsize=(20,10))
  plot_tree(spre_tree, filled=True, feature_names=['depth', 'precision', 'rate'], class_names=[str(i) for i in range(1, 6)])
  plt.show()
  
  spre_pruned_accuracy = spre_tree.score(X_test, y_test)
  print(f"Simple Pre-pruned Decision Tree Accuracy: {spre_pruned_accuracy:.4f}")
  
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.model_selection import GridSearchCV
  from sklearn.metrics import accuracy_score
  
  # Define parameters for Grid Search
  parameters = {
      'criterion': ['entropy', 'gini'],
      'splitter': ['best', 'random'],
      'max_depth': [None, 1, 2, 3, 4, 5],
      'max_features': ['sqrt', 'log2']
  }
  
  # Initialize the DecisionTreeClassifier
  clf = DecisionTreeClassifier(random_state=42)
  
  # Setup GridSearchCV
  hpre_tree = GridSearchCV(clf, param_grid=parameters, cv=5)
  
  # Fit GridSearchCV
  hpre_tree.fit(X_train, y_train)
  
  # Print best parameters and best score
  print("Best Parameters found: ", hpre_tree.best_params_)
  print("Best Score found: ", hpre_tree.best_score_)
  
  # Use the best estimator found by GridSearchCV to make predictions
  best_model = hpre_tree.best_estimator_
  y_pred = best_model.predict(X_test)
  
  # Evaluate the best model if needed
  accuracy = accuracy_score(y_test, y_pred)
  print(f"Accuracy of the best model: {accuracy:.4f}")
  
  # Display the tree structure
  plt.figure(figsize=(20,10))
  plot_tree(hpre_tree.best_estimator_, filled=True, feature_names=['depth', 'precision', 'rate'], class_names=[str(i) for i in range(1, 6)])
  plt.show()
  `,
      language: 'python',
    },
  };
  
  export default codeSections;
  