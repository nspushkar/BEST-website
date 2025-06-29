import React from 'react';
import Img1 from './imgs/image1.png';

const Lab2 = ({ onLinkClick }) => (
  <div className="p-6">
    <h1 className="text-2xl font-bold mb-4">
      <a href="#full" className="link-primary" onClick={() => onLinkClick('full')}>Simple Expert System for Decision Support</a>
    </h1>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#overview" className="link-primary" onClick={() => onLinkClick('full')}>Overview</a>
    </h2>
    <p className="mb-4">
      This project is a web-based application for diagnosing Bajaj vehicle issues using an expert system and Flask. 
      The application allows users to input symptoms their vehicle is experiencing, and based on predefined rules, 
      it provides a diagnosis.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#expertSystem" className="link-primary" onClick={() => onLinkClick('full')}>Expert System</a>
    </h2>
    <p className="mb-4">
      <strong>Definition:</strong> An expert system is a computer program that mimics the decision-making abilities of a human expert. 
      It uses predefined rules and a knowledge base to diagnose problems or provide solutions.
    </p>
    <img src={Img1} alt="Expert System Components" className="w-4/5 mx-auto my-4 p-2" />

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#components" className="link-primary" onClick={() => onLinkClick('Knowledge')}>Components of Expert System</a>
    </h2>
    <ol className="mb-4">
      <li className="mb-2">
        <a href="#knowledge" className="link-primary" onClick={() => onLinkClick('Knowledge')}>Knowledge Base</a>: 
        Contains domain-specific knowledge, in this case, rules for diagnosing issues with Bajaj vehicles.
      </li>
      <li className="mb-2">
        <a href="#inference" className="link-primary" onClick={() => onLinkClick('infrence')}>Inference Engine</a>: 
        Applies logical rules to the knowledge base to deduce conclusions from the given facts (symptoms).
      </li>
      <li className="mb-2">
        <a href="#ui" className="link-primary" onClick={() => onLinkClick('full')}>User Interface</a>: 
        Allows users to interact with the system, input their symptoms, and receive diagnoses.
      </li>
    </ol>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#flask" className="link-primary" onClick={() => onLinkClick('flask')}>Web Development with Flask</a>
    </h2>
    <p className="mb-4">
      <strong>Definition:</strong> Flask is a lightweight web framework for Python. It allows developers 
      to create web applications quickly and with minimal code.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#full" className="link-primary" onClick={() => onLinkClick('full')}>Components Used in Code</a>
    </h2>
    <ol className="mb-4">
      <li className="mb-2">
        <a href="#flaskApp" className="link-primary" onClick={() => onLinkClick('flask')}>Flask Application</a>: 
        The Flask object is initialized to create the application. Routes are defined using decorators 
        to handle different URL endpoints.
      </li>
      <li className="mb-2">
        <a href="#routes" className="link-primary" onClick={() => onLinkClick('routes')}>Routes</a>: 
        The Home Route ('/') handles both GET and POST requests. The GET request displays a form 
        where users can select symptoms, and the POST request processes the form data and displays the diagnosis.
      </li>
      <li className="mb-2">
        <a href="#htmlTemplate" className="link-primary" onClick={() => onLinkClick('html')}>HTML Template</a>: 
        Rendered using render_template_string, contains forms, checkboxes for symptoms, and buttons for submission. 
        Displays the diagnosis result or the form based on the request type.
      </li>
      <li className="mb-2">
        <a href="#runApp" className="link-primary" onClick={() => onLinkClick('run')}>Running the Application</a>: 
        Runs the Flask application in debug mode.
      </li>
    </ol>
  </div>
);

export default Lab2;
