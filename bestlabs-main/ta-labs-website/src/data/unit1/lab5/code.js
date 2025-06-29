const codeSections = {
    importLibraries: {
      code: `import numpy as np
  import pandas as pd
  import tensorflow as tf
  from tensorflow import keras
  from sklearn.model_selection import train_test_split
  import matplotlib.pyplot as plt 
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
  from sklearn.metrics import classification_report, accuracy_score`,
      language: 'python',
    },
    LoadData: {
      code: `train = pd.read_csv("./archive (3)/fashion-mnist_train.csv")
  test = pd.read_csv("./archive (3)/fashion-mnist_test.csv")`,
      language: 'python',
    },
    explore: {
      code: `train.head()
  
  test.head()`,
      language: 'python',
    },
    labels: {
      code: `# The categories of clothing in the dataset
  class_labels= ["T-shirt/top","Trouser","Pullover" ,"Dress","Coat" ,"Sandal" ,"Shirt" ,"Sneaker" ,"Bag" ,"Ankle boot"]`,
      language: 'python',
    },
    preprocess: {
      code: `x_train = train_data[:, 1:]
  y_train = train_data[:, 0]
  
  x_test = test_data[:, 1:]
  y_test = test_data[:, 0]
  
  # Normalize pixel values between 0 and 1 
  x_train = x_train / 255
  x_test = x_test / 255`,
      language: 'python',
    },
    SplitData: {
      code: `from sklearn.model_selection import train_test_split
  
  x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
  
  x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
  x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
  x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
  
  import matplotlib.pyplot as plt 
  
  plt.imshow(x_train[1])
  print(y_train[1])`,
      language: 'python',
    },
    DeepLearningModel: {
      code: `from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
  
  model = Sequential([
      Conv2D(filters=32, kernel_size=3, strides=(1,1), padding='valid', activation='relu', input_shape=(28, 28, 1)),
      MaxPooling2D(pool_size=(2,2)),
      Conv2D(filters=64, kernel_size=3, strides=(2,2), padding='same', activation='relu'),
      MaxPooling2D(pool_size=(2,2)),
      Flatten(),
      Dense(128, activation='relu'),
      Dropout(0.25),
      Dense(256, activation='relu'),
      Dropout(0.25),
      Dense(128, activation='relu'),
      Dense(10, activation='softmax')  
  ])
  
  model.compile(optimizer='adam', 
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])`,
      language: 'python',
    },
    TrainModel: {
      code: `model.fit(x_train, y_train, epochs=20, batch_size=16, verbose=1, validation_data=(x_val, y_val))`,
      language: 'python',
    },
    TestModel: {
      code: `y_pred = model.predict(x_test)
  
  model.evaluate(x_test, y_test)
  
  # Plot the training and validation accuracy and loss
  import matplotlib.pyplot as plt
  
  plt.figure(figsize=(16,30))
  j=1
  
  for i in np.random.randint(0, 1000, 60):
      plt.subplot(10, 6, j)
      j += 1
      plt.imshow(x_test[i].reshape(28, 28), cmap='Greys')
      plt.title('Actual = {} / {} \\nPredicted = {} / {}'.format(class_labels[int(y_test[i])], int(y_test[i]), class_labels[np.argmax(y_pred[i])], np.argmax(y_pred[i])))
      plt.axis('off')`,
      language: 'python',
    },
    Evaluate: {
      code: `from sklearn.metrics import classification_report, accuracy_score
  
  # Convert one-hot encoded labels to class indices if needed
  if y_test.ndim == 2 and y_test.shape[1] > 1:
      y_test_indices = np.argmax(y_test, axis=1)
  else:
      y_test_indices = y_test  # If y_test is already in integer form
  
  # Convert y_pred to class indices
  y_pred_indices = np.argmax(y_pred, axis=1)
  
  # Generate the classification report
  cr = classification_report(y_test_indices, y_pred_indices, target_names=class_labels)
  print(cr)
  
  # Calculate and print the overall accuracy
  accuracy = accuracy_score(y_test_indices, y_pred_indices)
  print(f"Overall Accuracy: {accuracy}")`,
      language: 'python',
    },
    SaveModel: {
      code: `# Save Model
  model.save('fashion_mnist_cnn_model.h5')
  
  # Load Model
  fashion_model = tf.keras.models.load_model('fashion_mnist_cnn_model.h5')`,
      language: 'python',
    },
    full: {
      code: `# Import Libraries
  
  import numpy as np
  import pandas as pd
  import tensorflow as tf
  from tensorflow import keras
  from sklearn.model_selection import train_test_split
  import matplotlib.pyplot as plt 
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
  from sklearn.metrics import classification_report, accuracy_score
  
  # Load Data
  
  train = pd.read_csv("./archive (3)/fashion-mnist_train.csv")
  test = pd.read_csv("./archive (3)/fashion-mnist_test.csv")
  
  train.head()
  
  # The categories of clothing in the dataset
  class_labels= ["T-shirt/top","Trouser","Pullover" ,"Dress","Coat" ,"Sandal" ,"Shirt" ,"Sneaker" ,"Bag" ,"Ankle boot"]
  
  test.head()
  
  # Store data as an array 
  train_data = np.array(train, dtype="float32")
  test_data = np.array(test, dtype="float32")
  
  x_train = train_data[:, 1:]
  y_train = train_data[:, 0]
  
  x_test = test_data[:, 1:]
  y_test = test_data[:, 0]
  
  # Normalize pixel values between 0 and 1 
  x_train = x_train / 255
  x_test = x_test / 255
  
  # Split Data
  
  x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
  
  x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
  x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
  x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
  
  plt.imshow(x_train[1])
  print(y_train[1])
  
  # Build a Deep Learning Model and Train the Model
  
  model = Sequential([
      Conv2D(filters=32, kernel_size=3, strides=(1,1), padding='valid', activation='relu', input_shape=(28, 28, 1)),
      MaxPooling2D(pool_size=(2,2)),
      Conv2D(filters=64, kernel_size=3, strides=(2,2), padding='same', activation='relu'),
      MaxPooling2D(pool_size=(2,2)),
      Flatten(),
      Dense(128, activation='relu'),
      Dropout(0.25),
      Dense(256, activation='relu'),
      Dropout(0.25),
      Dense(128, activation='relu'),
      Dense(10, activation='softmax')  
  ])
  
  model.compile(optimizer='adam', 
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
  
  model.fit(x_train, y_train, epochs=20, batch_size=16, verbose=1, validation_data=(x_val, y_val))
  
  # Test Model
  
  y_pred = model.predict(x_test)
  
  model.evaluate(x_test, y_test)
  
  # Plot the training and validation accuracy and loss
  import matplotlib.pyplot as plt
  
  plt.figure(figsize=(16,30))
  j=1
  
  for i in np.random.randint(0, 1000, 60):
      plt.subplot(10,6, j)
      j += 1
      plt.imshow(x_test[i].reshape(28,28), cmap='Greys')
      plt.title('Actual = {} / {} \\nPredicted = {} / {}'.format(class_labels[int(y_test[i])], int(y_test[i]), class_labels[np.argmax(y_pred[i])], np.argmax(y_pred[i])))
      plt.axis('off')
  
  # Evaluate 
  
  # Convert one-hot encoded labels to class indices if needed
  if y_test.ndim == 2 and y_test.shape[1] > 1:
      y_test_indices = np.argmax(y_test, axis=1)
  else:
      y_test_indices = y_test  # If y_test is already in integer form
  
  # Convert y_pred to class indices
  y_pred_indices = np.argmax(y_pred, axis=1)
  
  # Generate the classification report
  cr = classification_report(y_test_indices, y_pred_indices, target_names=class_labels)
  print(cr)
  
  # Calculate and print the overall accuracy
  accuracy = accuracy_score(y_test_indices, y_pred_indices)
  print(f"Overall Accuracy: {accuracy}")
  
  # Save Model
  model.save('fashion_mnist_cnn_model.h5')
  
  # Load Model
  
  fashion_model = tf.keras.models.load_model('fashion_mnist_cnn_model.h5')`,
      language: 'python',
    },
  };
  
  export default codeSections;
  