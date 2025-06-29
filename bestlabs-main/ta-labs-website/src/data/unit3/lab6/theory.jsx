import React from 'react';
import Img1 from './imgs/image1.png';
import Img2 from './imgs/image2.png';
import Img3 from './imgs/image3.png';

const Theory = ({ onLinkClick }) => (
  <div className="p-6">
    <h1 className="text-2xl font-bold mb-4">
      <a href="#Step1" className="link-primary text-blue-600" onClick={() => onLinkClick('Step1')}>
        Industrial Equipment Monitoring System
      </a>
    </h1>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#Step1" className="link-primary" onClick={() => onLinkClick('Step1')}>
        Step 1: Generate Training Data
      </a>
    </h2>
    <p className="mb-4">
      Synthetic sensor readings are generated to simulate temperature, pressure, and vibration from industrial equipment. Data points are classified as 'healthy' or 'unhealthy' based on defined conditions.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#Step2" className="link-primary" onClick={() => onLinkClick('Step2')}>
        Step 2: Save Training Data
      </a>
    </h2>
    <p className="mb-4">
      Data is saved to a CSV file named 'training_data.csv'.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#Step3" className="link-primary" onClick={() => onLinkClick('Step3')}>
        Step 3: Load and Preprocess Data
      </a>
    </h2>
    <p className="mb-4">
      The code loads the training data, converting the 'status' column into numerical values.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#Step4" className="link-primary" onClick={() => onLinkClick('Step4')}>
        Step 4: Split Data into Training and Testing Sets
      </a>
    </h2>
    <p className="mb-4">
      The data is split using the train_test_split function, setting aside 20% for testing.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#Step5" className="link-primary" onClick={() => onLinkClick('Step5')}>
        Step 5: Train the Model
      </a>
    </h2>
    <p className="mb-4">
      A Random Forest classifier is trained with parameters set for 100 trees and a random state of 42.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#Step6" className="link-primary" onClick={() => onLinkClick('Step6')}>
        Step 6: Evaluate the Model
      </a>
    </h2>
    <p className="mb-4">
      Model performance is assessed on the testing data with accuracy metrics.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#Step7" className="link-primary" onClick={() => onLinkClick('Step7')}>
        Step 7: Save the Model
      </a>
    </h2>
    <p className="mb-4">
      The trained model is saved to 'equipment_monitoring_model.pkl' using joblib.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#Step8" className="link-primary" onClick={() => onLinkClick('Step8')}>
        Step 8: Create the Dashboard
      </a>
    </h2>
    <p className="mb-4">
      A real-time dashboard is set up for monitoring, featuring graphs and a status indicator that updates based on the model's predictions.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#Step9" className="link-primary" onClick={() => onLinkClick('Step9')}>
        Step 9: Update Data
      </a>
    </h2>
    <p className="mb-4">
      Data is continuously updated in real-time, simulating sensor readings.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#Step10" className="link-primary" onClick={() => onLinkClick('Step10')}>
        Step 10: Update Graphs
      </a>
    </h2>
    <p className="mb-4">
      Graphs for temperature, pressure, and vibration are updated every second.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#Step11" className="link-primary" onClick={() => onLinkClick('Step11')}>
        Step 11: Update Status Indicator
      </a>
    </h2>
    <p className="mb-4">
      A status indicator updates in real-time to reflect the equipment's condition as 'healthy' or 'unhealthy'.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#Step12" className="link-primary" onClick={() => onLinkClick('Step12')}>
        Step 12: Run the Dashboard
      </a>
    </h2>
    <p className="mb-4">
      The dashboard runs, displaying real-time updates of the equipment's status.
    </p>

    <p className="mb-4">Here are some pictures of the graph when you run them:</p>
    <img src={Img1} alt="image1" className="w-4/5 mx-auto my-4 p-2" />
    <img src={Img2} alt="image2" className="w-4/5 mx-auto my-4 p-2" />
    <img src={Img3} alt="image3" className="w-4/5 mx-auto my-4 p-2" />
  </div>
);

export default Theory;
