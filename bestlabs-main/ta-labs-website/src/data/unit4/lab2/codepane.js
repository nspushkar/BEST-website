// CodePane.js
import React from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { solarizedlight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import codeSnippets from './code.js';

function CodePane({ selectedSnippet }) {
  // If no snippet is selected, return null or a message
  if (!selectedSnippet) {
    return <div>Select a snippet to view its code</div>;
  }

  const snippet = codeSnippets[selectedSnippet];
  // If the snippet does not exist, return a message
  if (!snippet) {
    return <div>Code snippet not found</div>;
  }

  const { code, language } = snippet;

  return (
    <div>
      <SyntaxHighlighter language={language} style={solarizedlight}>
        {code}
      </SyntaxHighlighter>
    </div>
  );
}

export default CodePane;