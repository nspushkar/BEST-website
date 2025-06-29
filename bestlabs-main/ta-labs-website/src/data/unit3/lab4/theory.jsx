import React from 'react';
import Img1 from './imgs/image1.png';
import Img2 from './imgs/image2.png';
import Img3 from './imgs/image3.png';
import Img4 from './imgs/image4.png';
import Img5 from './imgs/image5.png';
import Img6 from './imgs/image6.png';
import Img7 from './imgs/image7.png';
import Img8 from './imgs/image8.png';
import Img9 from './imgs/image9.png';
import Img11 from './imgs/image11.png';
import Img12 from './imgs/image12.png';
import Img13 from './imgs/image13.png';
import Img14 from './imgs/image14.png';
import Img15 from './imgs/image15.png';

const Theory = ({ onLinkClick }) => (
  <div className="p-6">
    <h1 className="text-2xl font-bold mb-4">
      <a href="#full" className="link-primary text-blue-600" onClick={() => onLinkClick('full')}>
        Building a Supervised Learning Model to Predict Machining Quality based on Operational Parameters
      </a>
    </h1>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#UnderstandingDecisionTrees" className="link-primary" onClick={() => onLinkClick('UnderstandingDecisionTrees')}>
        Understanding Decision Trees
      </a>
    </h2>
    <p className="mb-4">
      A decision tree is a popular machine learning algorithm used for classification and regression tasks. It is a tree-like model of decisions and their possible consequences. Here’s a simple breakdown of how it works:
    </p>
    <ul>
      <li><b>Root Node:</b> This is the topmost node in the tree, representing the entire dataset.</li>
      <li><b>Decision Nodes:</b> These are nodes where the dataset is split into different subsets based on certain conditions.</li>
      <li><b>Leaf Nodes:</b> These are terminal nodes that represent the final decision or output.</li>
    </ul>
    <p className="mb-4">
      Each internal node of the tree represents a "test" or "decision" on an attribute (e.g., "Is depth {">"} 50?"), and each branch represents the outcome of that decision. The leaves represent the class labels or regression values.
    </p>
    <img src={Img1} alt="image1" className="w-4/5 mx-auto my-4 p-2" />
    <img src={Img2} alt="image2" className="w-4/5 mx-auto my-4 p-2" />

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#GeneratingDataset" className="link-primary" onClick={() => onLinkClick('GeneratingDataset')}>
        Generating Dataset and Handling the Data
      </a>
    </h2>
    <p className="mb-4">Generated Dataset:</p>
    <p className="mb-4">depth precision rate</p>
    <ol className="ml-6">
      <li>38.079472 0.261706 185.947796</li>
      <li>95.120716 0.246979 542.359046</li>
      <li>73.467400 0.906255 873.072890</li>
      <li>60.267190 0.249546 732.492662</li>
      <li>16.445845 0.271950 806.754587</li>
    </ol>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#BuildingDecisionTree" className="link-primary" onClick={() => onLinkClick('BuildingDecisionTree')}>
        Building a Decision Tree
      </a>
    </h2>
    <p className="mb-4">
      When using scikit-learn to build a decision tree for classification, the algorithm splits the data at each node in a way that maximizes the "purity" of the resulting subsets. Gini impurity is one of the metrics used to measure this purity.
    </p>
    <ol className="ml-6">
      <li><b>Initialize the Root Node:</b> The process starts with the entire dataset at the root node.</li>
      <li><b>Calculate Gini Impurity:</b> For each possible split, calculate the Gini impurity of the subsets resulting from the split.</li>
      <img src={Img3} alt="image3" className="w-4/5 mx-auto my-4 p-2" />
      <li><b>Evaluate Splits:</b> Divide the dataset based on each feature and split value, calculating the weighted Gini impurity.</li>
      <li><b>Choose the Best Split:</b> The algorithm selects the split that results in the lowest weighted Gini impurity and uses this as the decision rule at the current node.</li>
      <li><b>Repeat for Child Nodes:</b> Continue recursively until a stopping criterion is met (maximum depth, minimum samples, node purity).</li>
      <li><b>Final Tree Structure:</b> The result is a tree structure where each internal node represents a decision based on a feature and a split value, and each leaf node represents a class label (for classification tasks).</li>
    </ol>
    <img src={Img4} alt="image4" className="w-4/5 mx-auto my-4 p-2" />
    <img src={Img5} alt="image5" className="w-4/5 mx-auto my-4 p-2" />
    <img src={Img6} alt="image6" className="w-4/5 mx-auto my-4 p-2" />

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#UnderstandingPruning" className="link-primary" onClick={() => onLinkClick('UnderstandingPruning')}>
        Understanding Pruning
      </a>
    </h2>
    <p className="mb-4">
      Pruning in the context of decision trees refers to the process of reducing the size of the tree by removing specific parts of it. This technique aims to improve the tree's ability to generalize to new, unseen data while avoiding overfitting to the training data.
    </p>
    <p className="mb-4">
      Pruning is necessary to prevent overfitting, where decision trees become overly complex and memorize noise or specifics of the training data, leading to poor performance on new data.
      We essentially prune by removing the nodes which have the least amount of information gain.
    </p>
    <img src={Img7} alt="image7" className="w-4/5 mx-auto my-4 p-2" />
    <img src={Img8} alt="image8" className="w-4/5 mx-auto my-4 p-2" />

    <p className="mb-4"><b>Types of Pruning:</b></p>
    <ul>
      <li><b>Cost Complexity Pruning (ccp):</b> Pruning in the context of decision trees refers to the process of reducing the size of the tree by removing specific parts of it. This technique aims to improve the tree's ability to generalize to new, unseen data while avoiding overfitting to the training data. Cost Complexity Pruning (ccp) balances tree complexity and training accuracy. Higher ccp_alpha values (𝛼) lead to more aggressive pruning, resulting in simpler trees with fewer nodes.</li>
      <img src={Img11} alt="image11" className="w-4/5 mx-auto my-4 p-2" />
      <img src={Img12} alt="image12" className="w-4/5 mx-auto my-4 p-2" />
      <img src={Img13} alt="image13" className="w-4/5 mx-auto my-4 p-2" />
      <li><b>Pre-pruning:</b> This involves setting stopping criteria before the tree is fully grown. It stops splitting nodes when further splitting does not lead to an improvement in model accuracy or when certain conditions are met.</li>
      <img src={Img14} alt="image14" className="w-4/5 mx-auto my-4 p-2" />
      <li><b>Post-Pruning (Reduced Error Pruning):</b> This technique involves growing the decision tree to its maximum size (fully grown) and then pruning back the nodes that do not provide significant improvements to the model's accuracy or validation performance.</li>
      <img src={Img15} alt="image15" className="w-4/5 mx-auto my-4 p-2" />
    </ul>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#ReadingEvaluatingTree" className="link-primary" onClick={() => onLinkClick('ReadingEvaluatingTree')}>
        Reading and Evaluating the Tree
      </a>
    </h2>
    <img src={Img9} alt="image9" className="w-4/5 mx-auto my-4 p-2" />
    <p className="mb-4">
      This is a representation of the decision rules the model has made. This allows us to easily see 
      how the algorithm makes each decision. It's just a representation of a decision tree in an easier to 
      read format as decision trees can become very complicated to pictorially represent and read.
    </p>
    <p className="mb-4">How to read:</p>
    <ul>
      <li>Each line represents a decision node in the tree. It specifies a condition based on a 
      feature.</li>
      <li>Indentation indicates the level of the node in the tree. More indentation means the node 
      is deeper in the tree.</li>
      <li>The condition (e.g., depth {"<"}= 50) is evaluated for each data point. If true, follow the 
      branch; if false, go to the next condition.
      </li>
    </ul>
  </div>
);

export default Theory;
