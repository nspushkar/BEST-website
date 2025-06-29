import React from 'react';
import cartypes from './imgs/cartypes.jpg'
import swipl from './imgs/swipl.png'

const Theory = ({ onLinkClick }) => (
  <div className="p-6">
    <h1 className="text-2xl font-bold mb-4">
      <a href="#vehicle_model_identification" className="link-primary" onClick={() => onLinkClick('full')}>
        Vehicle Model Identification
      </a>
    </h1>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#introduction" className="link-primary" onClick={() => onLinkClick('full')}>
        Introduction to Vehicle Model Identification
      </a>
    </h2>
    <p className="mb-4">
      This lab uses Prolog to identify vehicle models based on a set of predefined characteristics. The knowledge base contains different vehicle models (e.g., sedan, SUV) and a set of characteristics associated with each. Using rules, we can infer a vehicle model by matching user-input characteristics with model definitions.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#knowledge_base" className="link-primary" onClick={() => onLinkClick('characteristics')}>
        Knowledge Base: Models and Characteristics
      </a>
    </h2>
    <p className="mb-4">
      The knowledge base includes definitions for several vehicle types, such as sedan, SUV, and sports car. Each model has unique characteristics, like the number of doors, drive type, cargo space, and performance level. Characteristics are defined separately and combined to form each model’s profile.
    </p>

    <img src={cartypes} alt="Types of cars" className="w-4/5 mx-auto my-4 p-2" />

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#rules" className="link-primary" onClick={() => onLinkClick('rules')}>
        Rules for Model Identification
      </a>
    </h2>
    <p className="mb-4">
      Rules define the relationships between vehicle models and their characteristics. For instance, a sedan may have four doors, front-wheel drive, and a luxury interior. These rules are used to infer the vehicle type based on user-provided attributes.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#decision_support" className="link-primary" onClick={() => onLinkClick('decision_support')}>
        Decision Support for Model Identification
      </a>
    </h2>
    <p className="mb-4">
      The system interacts with users to identify the vehicle model based on responses to questions about the vehicle’s characteristics. This user-driven approach helps refine model identification by confirming or denying specific traits.
    </p>
    <img src={swipl} alt="Execution example" className="w-4/5 mx-auto my-4 p-2" />
  </div>
);

export default Theory;
