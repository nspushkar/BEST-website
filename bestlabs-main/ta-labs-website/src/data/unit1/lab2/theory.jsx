import React from 'react';
import grayScale1 from './imgs/grayScale1.png';
import grayScale2 from './imgs/grayScale2.png';
import gaussianBlur1 from './imgs/gaussianBlur1.png';
import gaussianBlur2 from './imgs/gaussianBlur2.png';
import gaussianBlur3 from './imgs/gaussianBlur3.png';
import canny1 from './imgs/canny1.png';
import contour1 from './imgs/contour1.png';
import shapeDetection from './imgs/shapeDetection.png';

const Theory = ({ onLinkClick }) => (
  <div className="p-6">
    <h1 className="text-2xl font-bold mb-4">
      <a href="#full" className="link-primary" onClick={() => onLinkClick('full')}>Shape Detection Theory Explained</a>
    </h1>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#grayscale" className="link-primary" onClick={() => onLinkClick('grayscale')}>Grayscale</a>
    </h2>
    <p className="mb-4">
      A grayscale image is one where the color information has been removed, leaving only shades of gray. Each pixel in a grayscale image represents an intensity value between 0 (black) and 255 (white).
    </p>
    <p className="mb-4">
      Converting to grayscale simplifies the image data, reducing the complexity and computational load for further processing like edge detection and contour detection. Working with a single intensity channel instead of three color channels (Red, Green, Blue) makes these operations more efficient.
    </p>
    <p className="mb-4">
      Achieved by taking a rough average of the R, G, B values.
    </p>
    <img src={grayScale1} alt="Grayscale Example 1" className="w-4/5 mx-auto my-4 p-2" />
    <img src={grayScale2} alt="Grayscale Example 2" className="w-4/5 mx-auto my-4 p-2" />

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#gaussianBlur" className="link-primary" onClick={() => onLinkClick('gaussianBlur')}>Gaussian Blur</a>
    </h2>
    <p className="mb-4">
      Gaussian blur is a smoothing filter that uses a Gaussian function to calculate the transformation to apply to each pixel in the image. It reduces noise and detail by averaging out the pixel values in a local neighborhood.
    </p>
    <p className="mb-4">
      Applying a Gaussian blur helps to reduce noise and minor variations in the image, which can improve the accuracy of edge detection. By smoothing the image, it becomes easier to detect significant edges and contours, as the algorithm won't be misled by small, irrelevant details.
    </p>
    <p className="mb-4">
      Gaussian blur is simply a method of blurring an image through the use of a Gaussian function. Below, you’ll see a 2D Gaussian distribution. Notice that there is a peak in the center and the curve flattens out as you move towards the edges.
    </p>
    <p className="mb-4">
      Imagine that this distribution is superimposed over a group of pixels in an image. It should be apparent looking at this graph, that if we took a weighted average of the pixel’s values and the height of the curve at that point, the pixels in the center of the group would contribute most significantly to the resulting value. This is, in essence, how Gaussian blur works.
    </p>
    <p className="mb-4">
      A Gaussian blur is applied by convolving the image with a Gaussian function. This concept of convolution will be explained more clearly in the upcoming lectures.
    </p>
    <img src={gaussianBlur1} alt="Gaussian Blur Example 1" className="w-4/5 mx-auto my-4 p-2" />
    <img src={gaussianBlur2} alt="Gaussian Blur Example 2" className="w-4/5 mx-auto my-4 p-2" />
    <img src={gaussianBlur3} alt="Gaussian Blur Example 3" className="w-4/5 mx-auto my-4 p-2" />

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#canny" className="link-primary" onClick={() => onLinkClick('canny')}>Edge Detection (Canny)</a>
    </h2>
    <p className="mb-4">
      Edge detection is a technique used to identify points in an image where the brightness changes sharply, indicating the presence of edges. The Canny edge detection algorithm is a popular method that involves several steps, including noise reduction, gradient calculation, non-maximum suppression, and edge tracking by hysteresis.
    </p>
    <p className="mb-4">
      Edges represent the boundaries of objects within an image. By detecting edges, the algorithm can identify and isolate the shapes present in the image. The Canny algorithm is effective because it can detect a wide range of edges in images.
    </p>
    <img src={canny1} alt="Canny Example" className="w-4/5 mx-auto my-4 p-2" />

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#contour" className="link-primary" onClick={() => onLinkClick('contourDetect')}>Contour Detection and Approximation</a>
    </h2>
    <p className="mb-4">
      Contours are useful for shape analysis and object detection and recognition. By finding contours, the algorithm can identify the outlines of shapes in the image, which is essential for further processing steps like shape approximation and classification. Uses edge detection as a precursor. The output is a list of contours, where each contour is a curve joining all the continuous points along a boundary with the same color or intensity.
    </p>
    <p className="mb-4">
      Contour approximation involves simplifying the contour shape by reducing the number of points on its perimeter while maintaining its overall structure.
    </p>
    <p className="mb-4">
      Approximating contours helps in identifying geometric shapes (like squares, circles) and objects within the image, making the analysis and recognition more efficient.
    </p>
    <img src={contour1} alt="Contour Detection Example" className="w-4/5 mx-auto my-4 p-2" />

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#shapeDetection" className="link-primary" onClick={() => onLinkClick('shape')}>Shape Detection</a>
    </h2>
    <p className="mb-4">
      Shape detection involves identifying and classifying different shapes within an image. It often follows contour detection, where shapes are analyzed based on their contours. Techniques used include the Hough transform for detecting lines and circles, and shape approximation for identifying polygons.
    </p>
    <p className="mb-4">
      Shape detection is crucial for tasks such as object recognition and scene understanding. By identifying the shapes present in an image, systems can classify and interpret the objects more effectively.
    </p>
    <img src={shapeDetection} alt="Shape Detection Example" className="w-4/5 mx-auto my-4 p-2" />
  </div>
);

export default Theory;
