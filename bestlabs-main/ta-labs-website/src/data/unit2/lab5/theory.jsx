import React from 'react';
import randomforest from './imgs/randomforest.png'

const Theory = ({ onLinkClick }) => (
  <div className="p-6">
    <h1 className="text-2xl font-bold mb-4">
      <a href="#random_forest" className="link-primary" onClick={() => onLinkClick('full')}>
        Unit 2 Lab 5: Random Forest Theory
      </a>
    </h1>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#introduction" className="link-primary" onClick={() => onLinkClick('full')}>
        Introduction to Random Forest
      </a>
    </h2>
    <p className="mb-4">
      Random Forest is an ensemble learning technique that builds multiple decision trees and merges them for better predictive accuracy. By using various trees, Random Forest reduces overfitting and improves generalization on unseen data.
    </p>

    <img src={randomforest} alt="Linear Regression Best fit line" className="w-4/5 mx-auto my-4 p-2" />

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#import_libraries" className="link-primary" onClick={() => onLinkClick('import_libraries')}>
        Import Libraries
      </a>
    </h2>
    <p className="mb-4">
      Essential libraries such as Pandas and NumPy are used for data manipulation. Sklearn provides modules for data splitting, scaling, model training, evaluation, and hyperparameter tuning with GridSearchCV.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#create_dataset" className="link-primary" onClick={() => onLinkClick('create_dataset')}>
        Create Dataset
      </a>
    </h2>
    <p className="mb-4">
      A synthetic dataset simulating sensor readings is generated, with each entry recording a timestamp, sensor reading, and a binary failure instance. This setup helps in simulating real-world sensor failure detection.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#preprocess_data" className="link-primary" onClick={() => onLinkClick('preprocess_data')}>
        Preprocess Data
      </a>
    </h2>
    <p className="mb-4">
      Data is split into training and testing sets, with sensor readings normalized to enhance model performance. Scaling ensures the model treats all features fairly without any one feature disproportionately influencing results.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#hyperparameter_tuning" className="link-primary" onClick={() => onLinkClick('hyperparameter_tuning')}>
        Hyperparameter Tuning and Model Training
      </a>
    </h2>
    <p className="mb-4">
      GridSearchCV optimizes parameters like the number of estimators, tree depth, and sample splitting criteria, ensuring a robust model configuration. This process improves model accuracy and adaptability.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#model_evaluation" className="link-primary" onClick={() => onLinkClick('model_evaluation')}>
        Evaluate the Model
      </a>
    </h2>
    <p className="mb-4">
      The model’s performance is evaluated using metrics like accuracy, confusion matrix, and classification report. These metrics provide insights into the model’s ability to detect sensor failures accurately.
    </p>
  </div>
);

export default Theory;
