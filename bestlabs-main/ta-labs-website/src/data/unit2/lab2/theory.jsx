import React from 'react';
import lrimg from './imgs/lr.png'

const Theory = ({ onLinkClick }) => (
  <div className="p-6">
    <h1 className="text-2xl font-bold mb-4">
      <a href="#overview" className="link-primary" onClick={() => onLinkClick('full')}>
        Prediction Model using Linear Regression
      </a>
    </h1>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#data_overview" className="link-primary" onClick={() => onLinkClick('data_overview')}>
        Step 1: Loading and Exploring the Data
      </a>
    </h2>
    <p className="mb-4">
      First, we load our dataset and perform basic exploratory data analysis (EDA) to understand feature distributions and check for missing values. 
      Visualizing data is essential for identifying patterns, outliers, or any inconsistencies that may impact the model.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#cleaning" className="link-primary" onClick={() => onLinkClick('preprocessing')}>
        Step 2: Data Cleaning and Preprocessing
      </a>
    </h2>
    <p className="mb-4">
      In data cleaning, we address missing values, handle categorical variables, and scale features as needed. Missing values can be filled 
      using the median or mean, and categorical features are encoded numerically to make them compatible with regression models.
      Scaling ensures that features with large ranges do not disproportionately affect model training.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#model_selection" className="link-primary" onClick={() => onLinkClick('model_selection')}>
        Step 3: Applying Linear Regression
      </a>
    </h2>
    <p className="mb-4">
      Linear regression models the relationship between features and a continuous target variable by fitting a linear equation.
      Coefficients represent the contribution of each feature. We evaluate the model's effectiveness using the R-squared value, which indicates 
      how well the model explains variability, and RMSE (Root Mean Squared Error), a measure of the model’s prediction accuracy.
    </p>

    <img src={lrimg} alt="Linear Regression Best fit line" className="w-4/5 mx-auto my-4 p-2" />

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#evaluation" className="link-primary" onClick={() => onLinkClick('evaluation')}>
        Step 4: Model Evaluation and Interpretation
      </a>
    </h2>
    <p className="mb-4">
      Model evaluation focuses on interpreting the linear regression coefficients and the goodness of fit metrics. 
      Positive or negative coefficients show the direction of the relationship with the target variable. 
      High R-squared values suggest a better fit, while a lower RMSE indicates improved accuracy in predictions.
    </p>
  </div>
);

export default Theory;
