import React from 'react';
import Img1 from './imgs/image1.jpg';
import vid1 from './imgs/video1.mp4';

const Lab5 = ({ onLinkClick }) => (
  <div className="p-6">
    <h1 className="text-2xl font-bold mb-4">
      <a href="#full" className="link-primary" onClick={() => onLinkClick('full')}>Lab Experiment: Voice-Controlled Servo Motor with Arduino</a>
    </h1>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#arduino_setup" className="link-primary" onClick={() => onLinkClick('arduino_code')}>Set Up Arduino IDE</a>
    </h2>
    <p className="mb-4">
      Download and install the Arduino IDE from <a href="https://www.arduino.cc/en/software" target="_blank" rel="noopener noreferrer">here</a>.
      Connect your Arduino board to your computer using a USB cable.
      Select the correct board and port in the Arduino IDE (`Tools &gt Board` and `Tools &gt Port`).
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#connect_servo" className="link-primary" onClick={() => onLinkClick('full')}>Connect the Servo Motor</a>
    </h2>
    <p className="mb-4">
      Connect the signal pin of the servo motor to digital pin 9 on the Arduino.
      Connect the power and ground pins of the servo motor to the 5V and GND pins on the Arduino, respectively.
    </p>
    <img src={Img1} alt="Servo Motor Connection" className="w-4/5 mx-auto my-4 p-2" />

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#upload_arduino_code" className="link-primary" onClick={() => onLinkClick('arduino_code')}>Upload Arduino Code</a>
    </h2>
    <p className="mb-4">
      Open the Arduino IDE and paste the provided code for controlling the servo motor. 
      Click the upload button (right arrow icon) in the Arduino IDE to upload the code to the Arduino board.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#install_libraries" className="link-primary" onClick={() => onLinkClick('full')}>Install Python Libraries</a>
    </h2>
    <p className="mb-4">
      Open a terminal or command prompt and run the following commands to install the necessary libraries:
      <code className="block my-4 p-2 bg-gray-200">pip install speechrecognition gtts pyserial</code>
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#write_python_code" className="link-primary" onClick={() => onLinkClick('python_code')}>Write Python Code</a>
    </h2>
    <p className="mb-4">
      Create a new Python file and paste the provided code for controlling the servo motor through voice commands.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#run_python_script" className="link-primary" onClick={() => onLinkClick('arduino_code')}>Run the Python Script</a>
    </h2>
    <p className="mb-4">
      Ensure the Arduino is connected to the computer and the code has been successfully uploaded. 
      Open a terminal or command prompt, navigate to the directory where your Python script is saved, and run the Python script by typing 
      <code>python your_script_name.py</code> and pressing Enter. 
      The Python script will start listening for voice commands and control the servo motor based on the recognized commands.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#conclusion" className="link-primary" onClick={() => onLinkClick('full')}>Conclusion</a>
    </h2>
    <p className="mb-4">
      This experiment demonstrates how to create a voice-controlled system using Python and Arduino to adjust a servo motor.
    </p>
    <video width="100%" height="50%" controls>
      <source src={vid1} type="video/mp4" />
      Your browser does not support the video tag.
    </video>
  </div>
);

export default Lab5;
