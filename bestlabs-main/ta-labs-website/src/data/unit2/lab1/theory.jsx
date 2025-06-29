import React from 'react';

const Theory = ({ onLinkClick }) => (
  <div className="p-6">
    <h1 className="text-2xl font-bold mb-4">
      <a href="#full" className="link-primary" onClick={() => onLinkClick('full')}>Prolog Querying: Theory Explained</a>
    </h1>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#install" className="link-primary" onClick={() => onLinkClick('install')}>Step 1: Installing Prolog</a>
    </h2>
    <p className="mb-4">
      To begin working with Prolog, you need to install the SWI-Prolog environment. Follow these steps:
    </p>
    <ol className="list-decimal pl-6 pt-4 text-gray-700">
      <li><strong>Download SWI-Prolog:</strong> Visit the official <a href="https://www.swi-prolog.org/download/stable" target="_blank" rel="noopener noreferrer" className="link-primary">SWI-Prolog website</a> and download the installer for your operating system (Windows, macOS, or Linux).</li><br />
      <li><strong>Run the Installer:</strong> Open the installer and follow the on-screen instructions to complete the installation. Make sure to check the option to <strong>add Prolog to PATH for all users</strong> so that you can run Prolog from the command line easily.</li><br />
      <li><strong>Verify Installation:</strong> After installing, open a terminal or command prompt and type <code>swipl</code> to check if Prolog is successfully installed.</li><br />
    </ol>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#createFile" className="link-primary" onClick={() => onLinkClick('createFile')}>Step 2: Creating a New Prolog File</a>
    </h2>
    <p className="mb-4">
      Once Prolog is installed, you can start writing Prolog code by creating a new `.pl` file:
    </p>
    <ol className="list-decimal pl-6 pt-4 text-gray-700">
      <li><strong>Open a Text Editor:</strong> Use a text editor (like VS Code, Sublime Text, or even Notepad) to create a new file with the `.pl` extension (e.g., `family.pl`).</li><br />
      <li><strong>Write Your Prolog Code:</strong> Inside the file, define facts, rules, and queries. See the code on the right for an example of how to write Prolog facts and rules.</li><br />
      <li><strong>Load the File in Prolog:</strong> In the terminal or command prompt, type <code>consult('path_to_file/family.pl')</code> to load the Prolog file into the interpreter.</li><br />
    </ol>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#step1" className="link-primary" onClick={() => onLinkClick('step1')}>Step 3: Defining Facts in Prolog</a>
    </h2>
    <p className="mb-4">
      In Prolog, facts represent simple, atomic truths about the world. A fact is a statement that declares something to be true, like a relationship or an attribute of an entity. For example:
    </p>
    <pre className="bg-gray-100 p-4 rounded-md text-gray-700">
      parent(john, mary).
    </pre>
    <p className="mb-4">
      This statement means "John is the parent of Mary." Facts are the building blocks of the knowledge base in Prolog.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#step2" className="link-primary" onClick={() => onLinkClick('step2')}>Step 4: Writing Rules in Prolog</a>
    </h2>
    <p className="mb-4">
      Rules in Prolog define logical relationships between facts. A rule is written with the syntax <code>:-</code>, meaning "if". For instance:
    </p>
    <pre className="bg-gray-100 p-4 rounded-md text-gray-700">
      sibling(X, Y) :- parent(Z, X), parent(Z, Y), X \= Y.
    </pre>
    <p className="mb-4">
      This rule defines that X and Y are siblings if they share the same parent Z, and X is not equal to Y (i.e., they are not the same person).
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#step3" className="link-primary" onClick={() => onLinkClick('step3')}>Step 5: Writing Queries in Prolog</a>
    </h2>
    <p className="mb-4">
      A query in Prolog is a question that you ask the Prolog interpreter based on the defined facts and rules. For example, to find out if John is the parent of Mary, you would query:
    </p>
    <pre className="bg-gray-100 p-4 rounded-md text-gray-700">
      ?- parent(john, mary).
    </pre>
    <p className="mb-4">
      Prolog will respond with "yes" if the statement is true based on the knowledge base, or "no" if it is not. 
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#step4" className="link-primary" onClick={() => onLinkClick('step4')}>Step 6: Experimenting with Queries</a>
    </h2>
    <p className="mb-4">
      After defining the facts and rules, you can experiment by writing queries to extract information. You can also modify the facts or rules to see how it affects the query outcomes.
    </p>
    <p className="mb-4">
      For example, adding a new fact like <code>parent(mary, alex)</code> allows you to query:
    </p>
    <pre className="bg-gray-100 p-4 rounded-md text-gray-700">
      ?- sibling(mary, alex).
    </pre>
    <p className="mb-4">This query checks if Mary and Alex are siblings based on the new fact.</p>
  </div>
);

export default Theory;
