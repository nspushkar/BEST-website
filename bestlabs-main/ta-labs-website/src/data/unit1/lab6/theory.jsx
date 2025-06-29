import React from 'react';
import Img1 from './imgs/image1.gif';
import Img2 from './imgs/image2.jpg';
import Img3 from './imgs/image3.gif';
import Img4 from './imgs/image4.jpg';
import Img5 from './imgs/image5.png';
import Img6 from './imgs/image6.webp';

const Theory = ({ onLinkClick }) => {
    return (
        <div className="theory-content">
            <h1 className="text-2xl font-bold mb-4">
                <a href="#full" className="link-primary" onClick={() => onLinkClick('full')}>Machine Learning Model Theories Explained</a>
            </h1>

            <p><strong>
                <a href="#logisticRegression" className="link-primary" onClick={() => onLinkClick("logisticRegression")}>Logistic Regression:</a>
            </strong></p>
            <ol>
                <li><b>Purpose:</b> Logistic regression is used to classify things into one of two groups.</li>
                <li><b>How it works:</b>
                    <ul>
                        <li><b>Input:</b> Takes various features (input variables).</li>
                        <li><b>Output:</b> Predicts the probability that an instance belongs to a certain class (between 0 and 1).</li>
                        <li><b>Function:</b> Uses a special function called the logistic (or sigmoid) function to convert inputs into a probability.</li>
                    </ul>
                </li>
                <li><b>Key Points:</b>
                    <ul>
                        <li>Uses examples with known outcomes to learn from.</li>
                        <li>It is a linear model, meaning the relationship between the input features and the output is linear</li>
                    </ul>
                </li>
            </ol>

            <img style={{ width: '100%' }} src={Img2} alt="image2" />
            <img style={{ width: '100%' }} src={Img3} alt="image3" />
            <img style={{ width: '100%' }} src={Img6} alt="image6" />

            <p><strong>
                <a href="#randomForest" className="link-primary" onClick={() => onLinkClick("randomForest")}>Random Forest:</a>
            </strong></p>
            <ol>
                <li><b>Purpose:</b> Random Forest is used for both classification (grouping things) and regression (predicting numbers).</li>
                <li><b>How it works:</b>
                    <ul>
                        <li><b>Trees:</b> Builds many decision trees to make predictions.</li>
                        <li><b>Combination:</b> Combines results from all trees for a final prediction.</li>
                        <li><b>Classification:</b> For classification, it chooses the most common class from all trees.</li>
                        <li><b>Regression:</b> For regression, it takes the average prediction from all trees.</li>
                    </ul>
                </li>
                <li><b>Key Points:</b>
                    <ul>
                        <li>Each tree is trained on a random subset of the data and uses a random subset of features.</li>
                        <li>Makes the model less likely to overfit and better at handling complex data. Hence provides better accuracy</li>
                    </ul>
                </li>
            </ol>

            <img style={{ width: '100%' }} src={Img1} alt="image1" />
            <img style={{ width: '100%' }} src={Img4} alt="image4" />
            <img style={{ width: '100%' }} src={Img5} alt="image5" />
        </div>
    );
};

export default Theory;
