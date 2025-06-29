// Define all your code snippets here with languages
const codeSnippets = {
    full: {
        code: `# Unit 2 Lab 2: Full Code for Data Processing and Model Implementation

# Step 1: Import Libraries and Load Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv('/path/to/dataset.csv')
print("First 5 rows of the dataset:")
print(data.head())

# Step 1: Dataset Overview
# Checking for missing values
missing_values = data.isnull().sum()
print("Missing Values per Column:")
print(missing_values)

# Visualize target distribution
plt.figure(figsize=(8, 6))
sns.histplot(data['target_variable'], bins=30)
plt.title('Target Variable Distribution')
plt.show()

# Step 2: Data Preprocessing
# Fill missing values
data.fillna(data.median(), inplace=True)

# Encoding categorical variables
encoder = LabelEncoder()
data['categorical_column'] = encoder.fit_transform(data['categorical_column'])

# Split data into features and target
X = data.drop(columns=['target_variable'])
y = data['target_variable']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Model Selection and Training
# Model: Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Alternative Model: Logistic Regression
# model = LogisticRegression(random_state=42)
# model.fit(X_train, y_train)

# Step 4: Model Evaluation
# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:")
print(report)
`,
        language: 'python'
    },

    overview: {
        code: `# Step 1: Import Libraries and Load Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv('/path/to/dataset.csv')
data.head()`,
        language: 'python'
    },

    data_overview: {
        code: `# Data Overview
# Checking for missing values
missing_values = data.isnull().sum()
print("Missing Values per Column:")
print(missing_values)

# Visualize target distribution
plt.figure(figsize=(8, 6))
sns.histplot(data['target_variable'], bins=30)
plt.title('Target Variable Distribution')
plt.show()`,
        language: 'python'
    },

    preprocessing: {
        code: `# Step 2: Data Preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Handling missing values (e.g., impute with median)
data.fillna(data.median(), inplace=True)

# Encoding categorical variables
encoder = LabelEncoder()
data['categorical_column'] = encoder.fit_transform(data['categorical_column'])

# Splitting data
X = data.drop(columns=['target_variable'])
y = data['target_variable']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)`,
        language: 'python'
    },

    model_selection: {
        code: `# Step 3: Model Selection and Training
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Choose and train model (example: Random Forest)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Alternatively, use Logistic Regression
# model = LogisticRegression(random_state=42)
# model.fit(X_train, y_train)`,
        language: 'python'
    },

    evaluation: {
        code: `# Step 4: Model Evaluation
from sklearn.metrics import accuracy_score, classification_report

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:")
print(report)`,
        language: 'python'
    }
};

export default codeSnippets;
