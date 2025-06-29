import React from 'react';
import Img1 from './imgs/image1.png';
import Img2 from './imgs/image2.png';
import Img3 from './imgs/image3.png';
import Img4 from './imgs/image4.png';
import Img6 from './imgs/image6.png';
import Img8 from './imgs/image8.gif';
import Img9 from './imgs/image9.png';
import Img10 from './imgs/image10.png';
import Img11 from './imgs/image11.png';
import Img12 from './imgs/image12.gif';
import Img13 from './imgs/image13.png';
import Img14 from './imgs/image14.gif';
import Img15 from './imgs/image15.jpg';

const Theory = ({ onLinkClick }) => {
    return (
        <div className="theory-content">
            <h1 className="text-2xl font-bold mb-4">
                <a href="#full" className="link-primary" onClick={() => onLinkClick('full')}>CNN Model for Image Classification Explained</a>
            </h1>

            <p><strong>
                <a href="#importLibraries" className="link-primary" onClick={() => onLinkClick("importLibraries")}>1. Import Libraries:</a>
            </strong></p>
            <p>To start, we need to import several essential libraries:</p>
            <ul>
                <li><b>NumPy:</b> Used for numerical computations.</li>
                <li><b>Pandas:</b> Used for data manipulation and analysis.</li>
                <li><b>TensorFlow and Keras:</b> Used for building and training neural networks.</li>
                <li><b>Scikit-learn (sklearn):</b> It provides many algorithms for tasks like classification, regression, and clustering.</li>
                <li><b>Matplotlib:</b> Matplotlib is a library in Python that helps create visualizations like graphs and charts.</li>
            </ul>

            <p><strong>
                <a href="#LoadData" className="link-primary" onClick={() => onLinkClick("LoadData")}>2. Load Data:</a>
            </strong></p>
            <p>We load our training and test datasets from CSV files using Pandas. The training data is stored in a variable called <b><i>train</i></b>, and the test data is stored in <b><i>test</i></b>.</p>

            <p><strong>
                <a href="#explore" className="link-primary" onClick={() => onLinkClick("explore")}>3. Explore Data</a>
            </strong></p>
            <p>To understand the structure and content of our data, we display the first 5 rows of the training dataset.</p>
            <img style={{ width: '100%' }} src={Img1} alt="image1" />
            <img style={{ width: '100%' }} src={Img2} alt="image2" />

            <p><strong>
                <a href="#labels" className="link-primary" onClick={() => onLinkClick("labels")}>4. Define Class Labels</a>
            </strong></p>
            <p>We define the class labels for the clothing items in our dataset. These labels represent the categories we aim to classify.</p>

            <p><strong>
                <a href="#preprocess" className="link-primary" onClick={() => onLinkClick("preprocess")}>5. Preprocess Data</a>
            </strong></p>
            <p>The data needs to be reshaped and normalized to be suitable for training our CNN model. Normalization ensures that all pixel values are between 0 and 1.</p>
            <img style={{ width: '100%' }} src={Img9} alt="image9" />

            <p><strong>
                <a href="#SplitData" className="link-primary" onClick={() => onLinkClick("SplitData")}>6. Split Data into Training and Validation Sets</a>
            </strong></p>
            <p>We split the training data into training and validation sets to evaluate the model's performance during training.</p>

            <p><strong>
                <a href="#DeepLearningModel" className="link-primary" onClick={() => onLinkClick("DeepLearningModel")}>7. Build CNN Model:</a>
            </strong></p>
            <p>We construct a Convolutional Neural Network (CNN) using Keras. The model consists of several layers, including convolutional layers, pooling layers, and dense layers.</p>
            <p><b>Convolutional Neural Networks (CNNs):</b> CNNs are specialized neural networks for processing data with a grid-like topology, such as images. They automatically and adaptively learn spatial hierarchies of features through backpropagation.</p>
            <img style={{ width: '100%' }} src={Img10} alt="image10" />

            <p style={{ marginLeft: '25px' }}><b>Convolutional layers:</b></p>
            <p style={{ marginLeft: '25px' }}>The core building block of a CNN is the convolutional layer. It applies filters to small regions of the input data, known as receptive fields. Each filter is a small matrix of weights that slides across the input data, performing a dot product with the input pixels. This process is known as convolution.</p>
            <img style={{ width: '100%' }} src={Img11} alt="image11" />
            <p>The output of the convolutional layer is a feature map, which is a two-dimensional representation of the input data. This feature map captures the presence of specific features in the input data, such as edges or lines.</p>
            <img style={{ width: '100%' }} src={Img12} alt="image12" />

            <p style={{ marginLeft: '25px' }}><b>Pooling layers:</b></p>
            <p style={{ marginLeft: '25px' }}>A pooling layer in a neural network helps simplify and reduce the size of the data from the convolutional layer. By doing this, it decreases the number of details the network needs to handle, which makes the network faster and more efficient.</p>
            <img style={{ width: '100%' }} src={Img6} alt="image6" />

            <p style={{ marginLeft: '25px' }}><b>Activation and Classification Layers:</b></p>
            <p style={{ marginLeft: '25px' }}>The output of the pooling layer is fed into an activation function, such as the Rectified Linear Unit (ReLU), which helps the network learn more complex features.</p>
            <p style={{ marginLeft: '25px' }}>The final layer of the CNN is typically a fully connected layer that outputs a probability distribution over all classes, allowing the network to classify the input data.</p>
            <img style={{ width: '100%' }} src={Img8} alt="image8" />
            <img style={{ width: '100%' }} src={Img13} alt="image13" />

            <p><strong>
                <a href="#TrainModel" className="link-primary" onClick={() => onLinkClick("TrainModel")}>8. Train Model:</a>
            </strong></p>
            <ul>
                <li>Feed the training images and their labels into the CNN.</li>
                <li>The CNN learns to recognize patterns and features in the images through a process called backpropagation.</li>
                <li>During training, the model's weights are adjusted to minimize the difference between its predictions and the true labels.</li>
            </ul>
            <img style={{ width: '100%' }} src={Img14} alt="image14" />

            <p><strong>
                <a href="#TestModel" className="link-primary" onClick={() => onLinkClick("TestModel")}>9. Test Model</a>
            </strong></p>
            <ul>
                <li>Feed the test images into the trained CNN model.</li>
                <li>Compare the model's predictions to the true labels to calculate metrics like accuracy, precision, and recall.</li>
            </ul>
            <img style={{ width: '100%' }} src={Img3} alt="image3" />

            <p><strong>
                <a href="#Evaluate" className="link-primary" onClick={() => onLinkClick("Evaluate")}>10. Evaluate the Model:</a>
            </strong></p>
            <p>After training, we evaluate the model on the validation set to see how well it performs.</p>
            <img style={{ width: '100%' }} src={Img15} alt="image15" />
            <img style={{ width: '100%' }} src={Img4} alt="image4" />

            <p><strong>
                <a href="#SaveModel" className="link-primary" onClick={() => onLinkClick("SaveModel")}>11. Save and Load the Model:</a>
            </strong></p>
            <p>The code saves the trained model to a file for future use. It can be loaded back into memory later to make predictions on new data.</p>
        </div>
    );
};

export default Theory;
