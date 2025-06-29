import React from 'react';
import Img1 from './imgs/image1.png';
import Img2 from './imgs/image2.png';
import Img3 from './imgs/image3.png';
import Img4 from './imgs/image4.png';

const Lab2 = ({ onLinkClick }) => (
  <div className="p-6">
    <h1 className="text-2xl font-bold mb-4">
      <a href="#full" className="link-primary" onClick={() => onLinkClick('full')}>Chatbot Using NLP Techniques for Customer Service or Operational Queries</a>
    </h1>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#spacy" className="link-primary" onClick={() => onLinkClick('SpaCy')}>Overview of spaCy</a>
    </h2>
    <p className="mb-4">
      spaCy is a robust library for Natural Language Processing (NLP) in Python. It is designed for high performance and large-scale text processing. The library is well-suited for real-world applications, providing tools for diverse NLP tasks.
    </p>
    <p className="mb-4">
      <strong>Key Features of spaCy:</strong>
      <ul className="list-disc ml-6">
        <li>Tokenization: Splits text into individual words and punctuation.</li>
        <li>Part-of-Speech Tagging: Identifies parts of speech for each word, like nouns and verbs.</li>
        <li>Named Entity Recognition (NER): Recognizes and classifies names, dates, and companies within text.</li>
        <li>Dependency Parsing: Establishes relationships between words, showing how sentences are structured.</li>
        <li>Text Classification: Trains models to categorize text into predefined labels.</li>
        <li>Pre-trained Models: Offers models trained on large datasets, ready to use for various languages.</li>
        <li>Customization: Allows customization of models for specific tasks and datasets.</li>
      </ul>
    </p>
    <img src={Img1} alt="1" className="w-4/5 mx-auto my-4 p-2" />

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#flask" className="link-primary" onClick={() => onLinkClick('Flask')}>Overview of Flask</a>
    </h2>
    <p className="mb-4">
      Flask is a minimalistic and flexible web framework for Python developers. It's easy to start with Flask for small projects and can also be expanded with various extensions for building complex applications.
    </p>
    <p className="mb-4">
      <strong>Key Features of Flask:</strong>
      <ul className="list-disc ml-6">
        <li>Micro-framework: Provides core functionality with options to include additional features as needed.</li>
        <li>Routing: Maps URLs to Python function handlers, making URL management easy.</li>
        <li>Templates: Integrates with Jinja2 for dynamic HTML rendering.</li>
        <li>Request Handling: Simplifies the management of incoming data and responses.</li>
        <li>Development Server: Includes a server for local testing and development.</li>
        <li>Extensions: Supports a wide range of plugins for added functionalities like ORM, form validation, and more.</li>
        <li>RESTful Support: Well-suited for creating APIs that can handle RESTful requests.</li>
      </ul>
    </p>
    <img src={Img2} alt="2" className="w-4/5 mx-auto my-4 p-2" />

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#restful" className="link-primary" onClick={() => onLinkClick('RESTful')}>Overview of RESTful APIs</a>
    </h2>
    <p className="mb-4">
      RESTful API is a design pattern for APIs. It stands for Representational State Transfer and uses HTTP methods for web services. It's widely adopted due to its simplicity and effectiveness in allowing various applications to communicate over the internet.
    </p>
    <p className="mb-4">
      <strong>Key Characteristics of RESTful APIs:</strong>
      <ul className="list-disc ml-6">
        <li>Stateless: Each request must have all necessary information; the server does not remember past requests.</li>
        <li>Client-Server Structure: Allows clients and servers to evolve separately without depending on each other.</li>
        <li>Cacheable: Clients can cache responses to improve performance and reduce server load.</li>
        <li>Uniform Interface: Makes the system simpler and more modular, allowing separate components to evolve.</li>
        <li>Hypermedia Driven: Clients interact with the server via hyperlinks provided dynamically by server responses.</li>
      </ul>
    </p>
    <img src={Img3} alt="3" className="w-4/5 mx-auto my-4 p-2" />

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#http_methods" className="link-primary" onClick={() => onLinkClick('RESTful')}>HTTP Methods in RESTful APIs</a>
    </h2>
    <p className="mb-4">
      <strong>HTTP Methods:</strong>
      <ul className="list-disc ml-6">
        <li>GET: Retrieves information from the server.</li>
        <li>POST: Sends new information to the server.</li>
        <li>PUT: Updates existing information on the server.</li>
        <li>DELETE: Removes existing information from the server.</li>
        <li>PATCH: Makes partial updates to existing information on the server.</li>
      </ul>
    </p>
    <img src={Img4} alt="4" className="w-4/5 mx-auto my-4 p-2" />

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#use_case" className="link-primary" onClick={() => onLinkClick('RESTful')}>Example Use Case of RESTful APIs</a>
    </h2>
    <p className="mb-4">
      A simple API for a book collection might include actions like retrieving all books, getting details of a specific book, adding a new book, updating an existing book, and deleting a book from the collection.
    </p>
    <p className="mb-4">
      <strong>The Key aspects of integrating these technologies are:</strong>
      <ul className="list-disc ml-6">
        <li><b>NLP with spaCy:</b> Utilizes spaCy for efficient text analysis and processing in web applications.</li>
        <li><b>Web Development with Flask:</b> Employs Flask's features to build user interfaces and manage web requests, facilitating interaction with NLP applications.</li>
        <li><b>RESTful API Design:</b> Develops APIs that are easy to use, maintain, and scale, enhancing communication between different software components.</li>
      </ul>
    </p>
  </div>
);

export default Lab2;
