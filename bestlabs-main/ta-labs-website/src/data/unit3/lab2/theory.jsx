import React from 'react';
import Img1 from './imgs/image1.png';
import Img2 from './imgs/image2.gif';

const LabTheory = ({ onLinkClick }) => {
  return (
    <div className="box3">
      <h1 className="text-2xl font-bold mb-4">
                <a href="#full" className="link-primary text-blue-600" onClick={() => onLinkClick('full')}>
                    Unsupervised Learning for Customer Segmentation
                </a>
            </h1>
      <p>
        <strong>
          <a href="#step1" className="link-primary" onClick={() => onLinkClick("Step1")}>Data Loading and Preprocessing</a>
        </strong>
      </p>
      <br />
      <ol style={{ marginLeft: '25px' }}>
        <li>
          <b>Loading Data:</b> The code loads customer data from a CSV file named 
          'Mall_Customers.csv' into a Pandas DataFrame called customer_data.
        </li>
        <br />

        <li>
          <b>Data Inspection:</b> The code checks the first five rows of the data using
          <ul>
            <li><b>customer_data.head()</b> the number of rows and columns using</li>
            <li><b>customer_data.shape</b>  and the data types of each column using</li>
            <li><b>customer_data.info()</b>  and the data types of each column using</li>
            <li><b>customer_data.isnull().sum()</b></li>
            <br />
          </ul>
        </li>
      </ol>
      <br />

      <p>
        <strong>
          <a href="#clustering" className="link-primary" onClick={() => onLinkClick("clustering")}>Clustering</a>
        </strong>
      </p>
      <br />
      <ol style={{ marginLeft: '25px' }}>
        <li>
          <strong>
            <a href="#clustering" className="link-primary" onClick={() => onLinkClick("clustering")}>K-Means Clustering:</a>
          </strong> The code performs K-Means clustering on the selected features 
          to identify customer segments. It calculates the Within Cluster Sum of Squares (WCSS) for 
          different numbers of clusters (from 1 to 10) and plots an elbow graph to determine the optimal 
          number of clusters.
        </li>
        <br />

        <img style={{ width: '100%' }} src={Img2} alt="image2" />
        <br /> <br />

        <li>
          <strong>
            <a href="#clustering" className="link-primary" onClick={() => onLinkClick("clustering")}>Optimal Number of Clusters:</a>
          </strong> The code determines the optimal number of clusters to be 5 based on the elbow graph.
        </li>
        <br />
        <li>
          <strong>
            <a href="#elbow" className="link-primary" onClick={() => onLinkClick("elbow")}>Elbow plot:</a>
          </strong> The code determines the optimal number of clusters to be 5 based on the elbow graph.
        </li>
        <p style={{ marginLeft: '25px' }}>
          The elbow method is a technique used to determine the optimal 
          number of clusters in K-Means clustering. It involves plotting the WCSS against the number 
          of clusters and identifying the point where the curve starts to flatten, indicating that 
          adding more clusters does not significantly improve the clustering.
        </p>
        <br />

        <img style={{ width: '100%' }} src={Img1} alt="image1" />
        <br /> <br />
        <li>
          <strong>
            <a href="#clustering" className="link-primary" onClick={() => onLinkClick("ClusterSummary")}>Clustering:</a>
          </strong> The code applies K-Means clustering with 5 clusters to the data and assigns each customer to a cluster.
        </li>
        <br />
      </ol>
      <br />

      <p>
        <strong>
          <a href="#visualisation" className="link-primary" onClick={() => onLinkClick("visualisation")}>Visualization</a>
        </strong>
      </p>
      <ol style={{ marginLeft: '25px' }}>
        <li>
          <strong>
            <a href="#ClusterSummary" className="link-primary" onClick={() => onLinkClick("ClusterSummary")}>Cluster Summary:</a>
          </strong> The code calculates the mean annual income and spending score for each cluster and prints the results.
        </li>
        <br />
        <li>
          <strong>
            <a href="#Gender" className="link-primary" onClick={() => onLinkClick("Gender")}>Gender Distribution:</a>
          </strong> The code counts the number of males and females in each cluster and prints the results.
        </li>
        <br />
        <li>
          <strong>
            <a href="#NoCustomer" className="link-primary" onClick={() => onLinkClick("NoCustomer")}>Number of Customers in Each Cluster:</a>
          </strong> The code visualizes the number of customers in each cluster using a count plot.
        </li>
        <br />
        <li>
          <strong>
            <a href="#OrgVisual" className="link-primary" onClick={() => onLinkClick("OrgVisual")}>Original Data Visualization:</a>
          </strong> The code visualizes the original data with each cluster represented by a different color.
        </li>
        <br />
        <li>
          <strong>
            <a href="#PcaK" className="link-primary" onClick={() => onLinkClick("PcaK")}>PCA and K-Means Clustering:</a>
          </strong> The code applies Principal Component Analysis (PCA) to 
          reduce the dimensionality of the data and then performs K-Means clustering on the transformed 
          data. It visualizes the clusters in the PCA space.
        </li>
      </ol>
    </div>
  );
};

export default LabTheory;
