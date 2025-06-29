import React from 'react';
import Img1 from './imgs/image1.png';
import Img2 from './imgs/image2.png';
import Img3 from './imgs/image3.png';
import Img4 from './imgs/image4.jpg';
import Img5 from './imgs/image5.jpg';
import Img6 from './imgs/image6.jpg';

const Theory = ({ onLinkClick }) => (
  <div className="p-6">
    <h1 className="text-2xl font-bold mb-4">
                <a href="#full" className="link-primary text-blue-600" onClick={() => onLinkClick('full')}>
                    Anomaly Detection and Real-time Monitoring on a Raspberri Pi
                </a>
            </h1>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#Section1" className="link-primary" onClick={() => onLinkClick('Section1')}>
        Section 1: Import Libraries and Initialize Video Capture
      </a>
    </h2>
    <p className="mb-4">
      Import necessary libraries for video processing, anomaly detection, and model handling. Initialize video capture to use the webcam.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#Section2" className="link-primary" onClick={() => onLinkClick('Section2')}>
        Section 2: Create Directories and Open CSV Files
      </a>
    </h2>
    <p className="mb-4">
      Create a directory to save frames where anomalies are detected. Open CSV files for logging anomalies and motion data.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#Section3" className="link-primary" onClick={() => onLinkClick('Section3')}>
        Section 3: Initialize Frames and Data Log
      </a>
    </h2>
    <p className="mb-4">
      Read the initial frames for motion detection. Initialize the list to store motion data. Set a flag for model readiness.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#Section4" className="link-primary" onClick={() => onLinkClick('Section4')}>
        Section 4: Motion Detection and Data Logging
      </a>
    </h2>
    <p className="mb-4">
      Detect motion between consecutive frames. Convert the difference to grayscale and apply Gaussian blur. Threshold and dilate the image to find contours representing motion. Draw rectangles around detected motion areas and log the motion data.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#Section5" className="link-primary" onClick={() => onLinkClick('Section5')}>
        Section 5: Initial Model Training and Retraining
      </a>
    </h2>
    <p className="mb-4">
      Train the initial Isolation Forest model after collecting 100 data points. Periodically retrain the model with new data every 50 frames.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#Section6" className="link-primary" onClick={() => onLinkClick('Section6')}>
        Section 6: Anomaly Detection and Logging
      </a>
    </h2>
    <p className="mb-4">
      Detect anomalies using the trained model. If an anomaly is detected, save the frame and log the anomaly.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#Section7" className="link-primary" onClick={() => onLinkClick('Section7')}>
        Section 7: Display Video and Clean Up
      </a>
    </h2>
    <p className="mb-4">
      Display the video feed with detected motion. Release video capture and close windows when 'q' is pressed.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#UnsupervisedLearning" className="link-primary" onClick={() => onLinkClick('UnsupervisedLearning')}>
        Unsupervised Learning and Isolation Forests
      </a>
    </h2>
    <ul>
      <li><strong>Unsupervised Learning:</strong> A type of machine learning that deals with data without predefined labels. The goal is to infer the natural structure within a dataset. In this project, we focus on anomaly detection using:</li>
    </ul>
    <img src={Img1} alt="image1" className="w-4/5 mx-auto my-4 p-2" />
    <ul>
      <li><strong>Isolation Forests:</strong> A popular and effective method for unsupervised anomaly detection. They work by isolating anomalies instead of profiling normal data points.</li>
    </ul>
    <p className="mb-4">Below are diagrams illustrating the concept of Isolation Trees and Isolation Forests, showing how multiple Isolation Trees are combined to detect anomalies.</p>
    <img src={Img3} alt="Isolation Trees" className="w-4/5 mx-auto my-4 p-2" />
    <img src={Img2} alt="Isolation Forests" className="w-4/5 mx-auto my-4 p-2" />

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#Implementation" className="link-primary" onClick={() => onLinkClick('Implementation')}>
        Implementation on Raspberry Pi
      </a>
    </h2>
    <p className="mb-4">This could also be implemented using a Raspberry Pi and a camera module instead of your webcam. That can be connected to the internet and be used for real-time monitoring of any environment.</p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      Images of Output
    </h2>
    <img src={Img4} alt="Output 1" className="w-4/5 mx-auto my-4 p-2" />
    <img src={Img5} alt="Output 2" className="w-4/5 mx-auto my-4 p-2" />
    <img src={Img6} alt="Output 3" className="w-4/5 mx-auto my-4 p-2" />
  </div>
);

export default Theory;
