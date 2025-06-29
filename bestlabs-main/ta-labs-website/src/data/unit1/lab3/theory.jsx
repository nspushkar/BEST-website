import React from 'react';
import Img1 from './imgs/image1.gif';
import Img2 from './imgs/image2.gif';
import Img4 from './imgs/image4.png';
import Img5 from './imgs/image5.png';

const Lab2 = ({ onLinkClick }) => (
  <div className="p-6">
    <h1 className="text-2xl font-bold mb-4">
      <a href="#full" className="link-primary" onClick={() => onLinkClick('full')}>Color Classification Theory Explained</a>
    </h1>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#step1" className="link-primary" onClick={() => onLinkClick('step1')}>Step 1: Defining Color Classification Rules</a>
    </h2>
    <p className="mb-4">
      This step involves defining rules to classify colors based on their HSV (Hue, Saturation, Value) values. The classification rules include:
    </p>
    <p className="mb-4">
      <strong>Red Objects:</strong> Classified as "Hot" <br />
      <strong>Blue Objects:</strong> Classified as "Cold" <br />
      <strong>Green Objects:</strong> Classified as "Natural" <br />
      <strong>Yellow Objects:</strong> Classified as "Warm" <br />
      <strong>Other Colors:</strong> Classified as "Unknown"
    </p>
    <img src={Img1} alt="1" className="w-4/5 mx-auto my-4 p-2" />
    <img src={Img2} alt="2" className="w-4/5 mx-auto my-4 p-2" />

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#color" className="link-primary" onClick={() => onLinkClick('color')}>Step 2: Using OpenCV to Capture Images and Detect Objects' Colors</a>
    </h2>
    <p className="mb-4">
      Use the provided Python code to capture an image from a webcam, detect the predominant color of an object in the image, and classify the object based on the predefined rules.
    </p>
    <p className="mb-4">
      Apply the rules to the detected objects and print out the classification.
    </p>
    <img src={Img4} alt="4" className="w-4/5 mx-auto my-4 p-2" />
    <img src={Img5} alt="5" className="w-4/5 mx-auto my-4 p-2" />

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#hsv" className="link-primary" onClick={() => onLinkClick('hsv')}>Step 3: Convert Image from BGR to HSV Color Space</a>
    </h2>
    <p className="mb-4">
      Converting the image from BGR (Blue, Green, Red) to HSV (Hue, Saturation, Value) color space is essential for accurate color classification.
    </p>
    <p className="mb-4">
      This conversion helps in analyzing colors more effectively by separating the color information (Hue) from the intensity (Value).
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#step3" className="link-primary" onClick={() => onLinkClick('step3')}>Step 4: Display Classification on Image</a>
    </h2>
    <p className="mb-4">
      Display the classification result on the image using OpenCV's `putText` function.
    </p>
    <p className="mb-4">
      Ensure the classification is visible and clearly indicates the detected color classification.
    </p>
    </div>
);

export default Lab2;
