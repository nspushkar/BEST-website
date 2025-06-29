import React from "react";
import Img1 from './imgs/image1.png';
import Img2 from './imgs/image2.png';
import Img3 from './imgs/image3.png';
import Img4 from './imgs/image4.png';
import Img5 from './imgs/image5.png';

const Theory = ({ onLinkClick }) => {
    return (
        <div>
            <h1 className="text-2xl font-bold mb-4">
                <a href="#full" className="link-primary text-blue-600" onClick={() => onLinkClick('full')}>
                    Image Classification Using Pre-trained VGG16 and Cosine Similarity
                </a>
            </h1>

            <p><strong><a href="#step1" className="link-primary" onClick={() => onLinkClick("Step1")}>1. Installing Libraries</a></strong></p>
            <p>The code starts by ensuring that all the necessary libraries, such as TensorFlow and scikit-learn, are installed to enable machine learning functionalities.</p> <br />

            <p><strong><a href="#importLibs" className="link-primary" onClick={() => onLinkClick("ImportLibs")}>2. Importing Libraries</a></strong></p>
            <p>Essential libraries like NumPy for handling data, TensorFlow for machine learning, and various scikit-learn modules for preprocessing and model evaluation are imported.</p> <br />

            <p><strong><a href="#definePath" className="link-primary" onClick={() => onLinkClick("DefinePath")}>3. Defining Dataset Path</a></strong></p>
            <p>The path where the dataset is stored is defined, guiding the code to locate and process the dataset correctly.</p> <br />

            <p><strong><a href="#loadDataset" className="link-primary" onClick={() => onLinkClick("LoadDataset")}>4. Loading Dataset</a></strong></p>
            <p>The dataset is loaded and prepared using TensorFlow's ImageDataGenerator, which also handles the image resizing and normalization.</p> <br />

            <p><strong><a href="#loadPrepData" className="link-primary" onClick={() => onLinkClick("LoadPrepData")}>5. Loading and Preparing Data</a></strong></p>
            <p>A function is set up to load the data, extract images and labels, and ensure that all the data needed for training and testing is ready and accessible.</p> <br />

            <p><strong><a href="#splitData" className="link-primary" onClick={() => onLinkClick("SplitData")}>6. Splitting Data into Training and Testing Sets</a></strong></p>
            <p>The data is divided into training and testing sets to provide a robust evaluation of the model's performance, maintaining an unbiased approach towards model validation.</p> <br />

            <p><strong><a href="#normalizeReshape" className="link-primary" onClick={() => onLinkClick("NormalizeReshape")}>7. Normalizing and Reshaping Features</a></strong></p>
            <p>Data normalization and reshaping are critical for preparing the data to fit the input requirements of the pre-trained VGG16 model, ensuring that each input is treated equally during model training.</p> <br />

            <p><strong><a href="#loadVGG16" className="link-primary" onClick={() => onLinkClick("LoadVGG16")}>8. Loading Pre-trained VGG16 Model</a></strong></p>
            <p>The pre-trained VGG16 model is loaded, set to extract deep features from the images. This model, trained on extensive datasets like ImageNet, provides a rich feature extraction capability.</p> <br />

            <p><strong><a href="#extractFeatures" className="link-primary" onClick={() => onLinkClick("ExtractFeatures")}>9. Extracting Features</a></strong></p>
            <p>A function is defined to pass images through the VGG16 model and extract significant features necessary for classifying the images effectively using machine learning techniques.</p> <br />

            <p><strong><a href="#computeCosSim" className="link-primary" onClick={() => onLinkClick("ComputeCosSim")}>10. Computing Cosine Similarity</a></strong></p>
            <p>Cosine similarity measures are computed between features of the training set and the test set to determine how similar the images are, aiding in the classification process based on the most similar training examples.</p> <br />

            <p><strong><a href="#evaluateClass" className="link-primary" onClick={() => onLinkClick("EvaluateClass")}>11. Evaluating Cosine Similarity Classification</a></strong></p>
            <p>The effectiveness of using cosine similarity for classification is evaluated, providing insights into the accuracy of this method in categorizing images based on learned patterns and similarities.</p> <br />

            <p><strong><a href="#preprocessImage" className="link-primary" onClick={() => onLinkClick("PreprocessImage")}>12. Preprocessing and Classifying New Images</a></strong></p>
            <p>Functions are prepared to preprocess new images, extract their features using the pre-trained model, and classify them by comparing these features to those of the images in the training set through cosine similarity.</p> <br />

            <p><strong><a href="#classifyImage" className="link-primary" onClick={() => onLinkClick("ClassifyImage")}>13. Displaying Classification Results</a></strong></p>
            <p>The classification results are displayed, showing the predicted labels for new images, which helps in understanding the model's performance and its practical application in real-world scenarios.</p> <br />

            <p>The Key aspects of this code are:-</p>
            <ul>
                <li><b>Feature Extraction:</b> Feature extraction involves identifying and extracting important characteristics or patterns from industrial equipment images. In the context of image classification, these features can include edges, textures, shapes, and more complex structures.</li> <br />
                <img style={{ width: '100%' }} src={Img1} alt="image1" /> <br /> <br />
                <img style={{ width: '100%' }} src={Img2} alt="image2" /> <br /> <br />
                <li><b>Cosine Similarity: </b> Cosine similarity measures the similarity between two vectors by comparing the angle between them. It is often used for image classification by comparing the feature vectors of images to determine how similar they are.</li>
                <img style={{ width: '100%' }} src={Img3} alt="image3" /> <br /> <br />
                <img style={{ width: '100%' }} src={Img4} alt="image4" /> <br /> <br />
                <img style={{ width: '100%' }} src={Img5} alt="image5" /> <br /> <br />
            </ul>
        </div>
    );
};

export default Theory;
