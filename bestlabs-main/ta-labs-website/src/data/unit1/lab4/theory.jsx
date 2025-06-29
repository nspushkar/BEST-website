import React from 'react';
import Img1 from './imgs/image1.png';
import Img3 from './imgs/image3.png';
import Img4 from './imgs/image4.png';
import Img5 from './imgs/image5.png';
import Img6 from './imgs/image6.gif';
import Img7 from './imgs/image7.png';
import Img8 from './imgs/image8.png';
import Img9 from './imgs/image9.png';
import Img10 from './imgs/image10.png';

const Theory = ({ onLinkClick }) => {
    return (
        <div className="theory-content">
            <h1 className="text-2xl font-bold mb-4">
      <a href="#full" className="link-primary" onClick={() => onLinkClick('full')}>Checking severity of damage using CNN</a>
    </h1>

            <p>
                <strong>
                    <a href="#step1" className="link-primary" onClick={() => onLinkClick("step1")}>
                        STEP 1: Defining Parameters for Training and Testing Phase
                    </a>
                </strong>
            </p>
            <img style={{width: '100%'}} src={Img3} alt="image3" /> 
            <img style={{width: '100%'}} src={Img4} alt="image4" /> 

            <p>
                <strong>
                    <a href="#step2" className="link-primary" onClick={() => onLinkClick("step2")}>
                        STEP 2: Defining the CNN to Predict the Severity of Damage on Vehicles
                    </a>
                </strong>
            </p>
            <p><b>Convolution layer:</b> The core building block of a CNN is the convolutional layer. It applies filters to small regions of the input data, known as receptive fields. Each filter is a small matrix of weights that slides across the input data, performing a dot product with the input pixels. This process is known as convolution.</p>
            <img style={{width: '100%'}} src={Img1} alt="image1" /> 
            <img style={{width: '100%'}} src={Img5} alt="image5" /> 
            <p>The output of the convolutional layer is a feature map, which is a two-dimensional representation of the input data. This feature map captures the presence of specific features in the input data, such as edges or lines.</p>
            <img style={{width: '100%'}} src={Img6} alt="image6" /> 
            <p><b>Pooling Layer:</b>The pooling layer simplifies the output of the convolutional layer by performing nonlinear downsampling. This reduces the number of parameters that the network needs to learn, making the network more efficient.</p>
            <img style={{width: '100%'}} src={Img7} alt="image7" /> 
            <p><b>Flatten Layer:</b></p>
            <p>The flatten layer is a component of the convolutional neural networks (CNN's). A complete convolutional neural network can be broken down into two parts:</p>
            <p><b>CNN:</b> The convolutional neural network that comprises the convolutional layers.</p>
            <p><b>ANN:</b>The artificial neural network that comprises dense layers.</p>
            <img style={{width: '100%'}} src={Img8} alt="image8" /> 
            <img style={{width: '100%'}} src={Img9} alt="image9" /> 

            <p>
                <strong>
                    <a href="#step3" className="link-primary" onClick={() => onLinkClick("step3")}>
                        Step 3: Using a Pretrained Model
                    </a>
                </strong>
            </p>
            <p>This code uses a pre-trained DenseNet169 model, which means it’s a model that has already been trained on a large dataset (ImageNet) and knows how to recognize many common features in images. We exclude the final layers of DenseNet169 to add our own custom layers for our specific task. This allows us to take advantage of the pre-trained model's knowledge and adapt it to classify images into the three categories.</p>
            <p>DenseNet (Densely Connected Convolutional Networks) connects each layer to every other layer in a feed-forward fashion. This creates a dense connection pattern where the input to a layer includes the feature maps of all preceding layers.DenseNet-169 specifically refers to a DenseNet model with 169 layers.</p>
            <p>Using this pre-trained model helps our new model learn faster and perform better with less training data and time. This approach is known as transfer learning.</p>
            <img style={{width: '100%'}} src={Img10} alt="image10" /> 
        </div>
    );
};

export default Theory;
