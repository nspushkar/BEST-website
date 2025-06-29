import React from 'react';
import Img1 from './imgs/image1.gif';
import Img2 from './imgs/image2.png';
import Img3 from './imgs/image3.png';

const Theory = ({ onLinkClick }) => (
  <div className="p-6">
    <h1 className="text-2xl font-bold mb-4">
      <a href="#full" className="link-primary text-blue-600" onClick={() => onLinkClick('full')}>
        Implementation of an Unsupervised Algorithm to Cluster Machine Sound for Anomaly Detection
      </a>
    </h1>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#full" className="link-primary text-blue-600" onClick={() => onLinkClick('full')}>
        DBSCAN: A Pictorial Representation and Explanation
      </a>
    </h2>
    <p className="mb-4">
      DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a popular clustering algorithm that groups together points that are closely packed and marks points that lie alone in low-density regions as outliers.
    </p>
    
    <h3 className="text-lg font-semibold mb-2">
      <a href="#full" className="link-primary text-blue-600" onClick={() => onLinkClick('full')}>
        Key Concepts
      </a>
    </h3>
    <ul className="mb-4">
      <li><strong>Epsilon (eps):</strong> The maximum distance between two points for them to be considered as in the same neighborhood.</li>
      <li><strong>MinPts (min_samples):</strong> The minimum number of points required to form a dense region (a cluster).</li>
    </ul>
    <img src={Img1} alt="DBSCAN illustration" className="w-4/5 mx-auto my-4 p-2" />

    <h3 className="text-lg font-semibold mb-2">
      <a href="#full" className="link-primary text-blue-600" onClick={() => onLinkClick('full')}>
        Working of DBSCAN
      </a>
    </h3>
    <ul className="mb-4">
      <li><strong>Core Points:</strong> A point is a core point if it has at least min_samples points (including itself) within eps.</li>
      <li><strong>Border Points:</strong> A point that has fewer than min_samples points within eps, but is in the neighborhood of a core point.</li>
      <li><strong>Noise Points:</strong> A point that is neither a core point nor a border point.</li>
    </ul>

    <h3 className="text-lg font-semibold mb-2">
      <a href="#full" className="link-primary text-blue-600" onClick={() => onLinkClick('full')}>
        Algorithm Steps
      </a>
    </h3>
    <ol className="ml-6 mb-4">
      <li><strong>Label Points:</strong>
        <ul className="ml-4">
          <li>If a point has at least min_samples points within eps, mark it as a core point.</li>
          <li>If a point is within eps distance of a core point, mark it as a border point.</li>
          <li>If a point is neither, mark it as noise.</li>
        </ul>
      </li>
      <li><strong>Cluster Formation:</strong>
        <ul className="ml-4">
          <li>Start with an arbitrary point and retrieve its eps-neighborhood.</li>
          <li>If it’s a core point, form a cluster. Add all points within eps as part of this cluster.</li>
          <li>Recursively visit each point within this cluster and include all reachable points within eps.</li>
          <li>Continue until all points are visited.</li>
        </ul>
      </li>
    </ol>

    <h3 className="text-lg font-semibold mb-2">
      <a href="#full" className="link-primary text-blue-600" onClick={() => onLinkClick('full')}>
        Advantages and Disadvantages
      </a>
    </h3>
    <ul className="mb-4">
      <li><strong>Advantages:</strong> DBSCAN can find arbitrarily shaped clusters and is robust to noise and outliers. It does not require specifying the number of clusters beforehand.</li>
      <img src={Img2} alt="Advantages of DBSCAN" className="w-4/5 mx-auto my-4 p-2" />
      <li><strong>Disadvantages:</strong> DBSCAN's performance depends on the choice of eps and min_samples. It may struggle with varying densities and high-dimensional data.</li>
      <img src={Img3} alt="Disadvantages of DBSCAN" className="w-4/5 mx-auto my-4 p-2" />
    </ul>
  </div>
);

export default Theory;
