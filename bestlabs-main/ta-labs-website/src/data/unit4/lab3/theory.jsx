import React from 'react';
import Img3 from './imgs/image3.png';

const Theory = ({ onLinkClick }) => (
  <div className="p-6">
    <h1 className="text-2xl font-bold mb-4">
      <a href="#full" className="link-primary" onClick={() => onLinkClick('full')}>Lab Experiment 3: Developing a Chatbot for Website FAQ</a>
    </h1>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#step1" className="link-primary" onClick={() => onLinkClick('Step1')}>Step 1: Install the Necessary Library</a>
    </h2>
    <p className="mb-4">
      We need to install the google-cloud-dialogflow library to interact with Dialogflow.
      Next, we need to import the required libraries and set up authentication using our JSON key file.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#step2" className="link-primary" onClick={() => onLinkClick('Step2')}>Step 2: Set Up Authentication</a>
    </h2>
    <p className="mb-4">
      We need to set an environment variable to point to our JSON key file. This file contains credentials for our Google Cloud project. A google account has already been created and its json key has been placed in the keyfile.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#step3" className="link-primary" onClick={() => onLinkClick('Step3')}>Step 3: Initialize Dialogflow Client</a>
    </h2>
    <p className="mb-4">
      We need to initialize the Dialogflow client to interact with Dialogflow and set our Project ID and Parent Path. Both have already been done in the code above.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#step4" className="link-primary" onClick={() => onLinkClick('Step4')}>Step 4: Define Helper Functions</a>
    </h2>
    <p className="mb-4">
      We need two helper functions: one to get existing intents and another to create or update intents.
    </p>
    <ul className="list-disc ml-6">
      <li><strong>Get Existing Intents:</strong> This function fetches all existing intents in the Dialogflow agent and returns them as a dictionary for quick lookup.</li>
      <li><strong>Create or Update an Intent:</strong> This function creates a new intent or updates an existing one with training phrases and response messages.</li>
    </ul>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#step5" className="link-primary" onClick={() => onLinkClick('Step5')}>Step 5: Prepare FAQs Data</a>
    </h2>
    <p className="mb-4">
      We need to prepare the data for our FAQs. This data will be used to create or update intents in Dialogflow when we call our helper functions. Use <code>get_existing_intents()</code> to fetch already existing intents and <code>create_or_update_intent()</code> to either create or update the intent.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#step6" className="link-primary" onClick={() => onLinkClick('Step6')}>Step 6: Define Function to Detect Intents</a>
    </h2>
    <p className="mb-4">
      We need a function to detect intents based on user queries. This function sends queries to Dialogflow and prints the responses.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#step7" className="link-primary" onClick={() => onLinkClick('Step7')}>Step 7: Test the Chatbot</a>
    </h2>
    <p className="mb-4">
      We define a list of test queries that correspond to the FAQs and call <code>detect_intent_texts()</code> to verify that the chatbot responds correctly. If you re-run this block you will get different versions of responses, which helps in improving user experience. Note--Rerun this block if detected intent is Default Fallback Intents.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#step8" className="link-primary" onClick={() => onLinkClick('Step8')}>Step 8: Training Phrases</a>
    </h2>
    <p className="mb-4">
      Training phrases are examples of how people ask questions to a chatbot. They teach the chatbot to understand and answer questions correctly. For example, if someone asks "When do you open?" or "What time do you start?", these examples help the chatbot learn what to say. Having lots of examples helps the chatbot get better at talking with people and giving the right answers, improving how well it can help users and handle different conversations.
    </p>
    <p className="mb-4">
      This code snippet will update the "What are your opening hours?" intent with additional training phrases and then test it with sample queries. Note--Rerun this block if detected intent is Default Fallback Intents.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#step9" className="link-primary" onClick={() => onLinkClick('Step9')}>Step 9: Deletion</a>
    </h2>
    <p className="mb-4">
      We can run this block if we intend to make new intents and remove the previous ones. In addition, the <code>delete_all_intents()</code> function ensures that you can start from a clean slate by deleting all existing intents in the Dialogflow agent. This is useful for resetting the agent during development and testing.
    </p>

    <img src={Img3} alt="3" className="w-4/5 mx-auto my-4 p-2" />
  </div>
);

export default Theory;
