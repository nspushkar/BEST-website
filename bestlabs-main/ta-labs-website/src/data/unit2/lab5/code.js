// Define all your code snippets here with languages
const codeSnippets = {
    full: {
        code: `# Full Random Forest Code for Lab 5

# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Create Synthetic Dataset
np.random.seed(42)
num_entries = 1000
timestamps = pd.date_range(start='2024-01-01', periods=num_entries, freq='H')
sensor_readings = np.random.uniform(low=0, high=100, size=num_entries)
failures = np.random.choice([0, 1], size=num_entries, p=[0.95, 0.05])

historical_sensor_data = pd.DataFrame({
    'Timestamp': timestamps,
    'SensorReading': sensor_readings,
    'Failure': failures
})

# Step 3: Preprocess Data
X = historical_sensor_data[['SensorReading']]
y = historical_sensor_data['Failure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Hyperparameter Tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 20],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 4],
    'class_weight': ['balanced']
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=2,
                           scoring='f1_macro',
                           verbose=1)
grid_search.fit(X_train_scaled, y_train)

# Step 5: Model Evaluation
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)
`,
        language: 'python'
    },

    import_libraries: {
        code: `# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix`,
        language: 'python'
    },

    create_dataset: {
        code: `# Step 2: Create Synthetic Dataset
np.random.seed(42)
num_entries = 1000
timestamps = pd.date_range(start='2024-01-01', periods=num_entries, freq='H')
sensor_readings = np.random.uniform(low=0, high=100, size=num_entries)
failures = np.random.choice([0, 1], size=num_entries, p=[0.95, 0.05])

historical_sensor_data = pd.DataFrame({
    'Timestamp': timestamps,
    'SensorReading': sensor_readings,
    'Failure': failures
})`,
        language: 'python'
    },

    preprocess_data: {
        code: `# Step 3: Preprocess Data
X = historical_sensor_data[['SensorReading']]
y = historical_sensor_data['Failure']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)`,
        language: 'python'
    },

    hyperparameter_tuning: {
        code: `# Step 4: Hyperparameter Tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 20],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 4],
    'class_weight': ['balanced']
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=2,
                           scoring='f1_macro',
                           verbose=1)
grid_search.fit(X_train_scaled, y_train)`,
        language: 'python'
    },

    model_evaluation: {
        code: `# Step 5: Model Evaluation
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)`,
        language: 'python'
    }
};

export default codeSnippets;
