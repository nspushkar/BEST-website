import React from 'react';
import Img2 from './imgs/image2.jpg';
import Img3 from './imgs/image3.jpg';
import Img4 from './imgs/image4.jpg';
import vid1 from './imgs/video1.mp4';

const Lab2 = ({ onLinkClick }) => (
  <div className="p-6">
    <h1 className="text-2xl font-bold mb-4">
      <a href="#full" className="link-primary" onClick={() => onLinkClick('full')}>Lab Experiment: Voice-Controlled LED with Arduino</a>
    </h1>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#arduino_setup" className="link-primary" onClick={() => onLinkClick('arduino_setup')}>Set Up Arduino IDE</a>
    </h2>
    <p className="mb-4">
      Download and install the Arduino IDE from <a href="https://www.arduino.cc/en/software" target="_blank" rel="noopener noreferrer">here</a>.
      Connect your Arduino board to your computer using a USB cable.
      Select the correct board and port in the Arduino IDE (`Tools &gt Board` and `Tools &gt Port`).
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#connect_led" className="link-primary" onClick={() => onLinkClick('full')}>Connect the LED</a>
    </h2>
    <p className="mb-4">
      Connect the anode (longer leg) of the LED to a 220-ohm resistor. Connect the other end of the resistor to digital pin 7 on the Arduino.
      Connect the cathode (shorter leg) of the LED to the ground (GND) pin on the Arduino.
    </p>
    <img src={Img3} alt="3" className="w-4/5 mx-auto my-4 p-2" />

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#install_libraries" className="link-primary" onClick={() => onLinkClick('microphone')}>Install Python Libraries and Set Up Microphone</a>
    </h2>
    <p className="mb-4">
      Open a terminal or command prompt and run the following commands:
      <code className="block my-4 p-2 bg-gray-200">pip install speechrecognition gtts pyserial</code>
      <b>Libraries Used:</b>
      <ul className="list-disc ml-6">
        <li><b>speech_recognition:</b> Captures and recognizes speech from the microphone.</li>
        <li><b>gTTS (Google Text-to-Speech):</b> Converts text to speech using Google's Text-to-Speech API.</li>
        <li><b>os:</b> Allows using operating system dependent functionality.</li>
        <li><b>serial:</b> Enables serial communication with the Arduino board.</li>
        <li><b>time:</b> Provides various time-related functions.</li>
      </ul>
    </p>
    <img src={Img4} alt="4" className="w-4/5 mx-auto my-4 p-2" />

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#microphone_input" className="link-primary" onClick={() => onLinkClick('microphone')}>Microphone Input</a>
    </h2>
    <p className="mb-4">
      The <code>sr.Microphone()</code> function captures audio input from the microphone.
      <code>adjust_for_ambient_noise</code> adjusts the recognizer sensitivity to ambient noise.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#speech_recognition" className="link-primary" onClick={() => onLinkClick('microphone')}>Speech Recognition Process</a>
    </h2>
    <p className="mb-4">
      The <code>recognizer.listen</code> function captures the audio from the microphone.
      The <code>recognizer.recognize_google</code> function sends the captured audio to Google’s speech recognition API,
      which uses advanced DNNs and probabilistic models to transcribe the speech to text.
    </p>
    <img src={Img4} alt="4" className="w-4/5 mx-auto my-4 p-2" />

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#arduino_code" className="link-primary" onClick={() => onLinkClick('arduino_code')}>Arduino Code and Serial Communication</a>
    </h2>
    <p className="mb-4">
      Open the Arduino IDE and paste the following code. Ensure the Arduino is connected to the computer and the code has been successfully uploaded.
      Click the upload button (right arrow icon) in the Arduino IDE to upload the code to the Arduino board.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#run_python" className="link-primary" onClick={() => onLinkClick('python_script')}>Run the Python Script</a>
    </h2>
    <p className="mb-4">
      Ensure the Arduino is connected to the computer and the code has been successfully uploaded. Open a terminal or command prompt, navigate to the directory where your Python script is saved, and run the Python script by typing <code>python your_script_name.py</code> and pressing Enter. The Python script will start listening for voice commands and control the LED based on the recognized commands.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#math_backing" className="link-primary" onClick={() => onLinkClick('full')}>Mathematical Backing for Speech Recognition</a>
    </h2>
    <p className="mb-4">
      The speech_recognition library uses Hidden Markov Models (HMMs) and Deep Neural Networks (DNNs) for recognizing speech.
      HMMs are used to model the sequence of speech sounds, while DNNs are used for acoustic modeling,
      converting audio signals into phonetic representations.
    </p>
    <img src={Img2} alt="2" className="w-4/5 mx-auto my-4 p-2" />

    <br />
    <p><b>Conclusion:</b> This experiment demonstrates how to create a voice-controlled system using Python and Arduino to switch an LED on and off.</p> <br />
    <video width="100%" height="50%" controls>
      <source src={vid1} type="video/mp4" />
      Your browser does not support the video tag.
    </video>
  </div>
);

export default Lab2;
