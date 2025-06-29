// CodePane.js
import React,{useState} from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { solarizedlight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import codeSnippets from './code.js';
import Editor from '@monaco-editor/react';

function CodePane({ selectedSnippet }) {



  const [code, setCode] = useState('');
  const [output, setOutput] = useState('');
  const [isRunning, setIsRunning] = useState(false);



  // If no snippet is selected, return null or a message
  // if (!selectedSnippet) {
  //   return <div>Select a snippet to view its code</div>;
  // }
   const snippet = codeSnippets[selectedSnippet];
  
  // // If the snippet does not exist, return a message
  // if (!snippet) {
  //   return <div>Code snippet not found</div>;
  // }





  React.useEffect(() => {
    if (snippet) {
      setCode(snippet.code);
    }
  }, [snippet]);

  const handleEditorChange = (value) => {
    setCode(value || '');
  };

  const runCode = async () => {
    setIsRunning(true);
    setOutput('');
  
    try {
      const response = await fetch('http://localhost:5050/run', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          language: snippet.language,
          code,
        }),
      });
  
      const data = await response.json();
      setOutput(data.output || data.error || 'No output');
    } catch (error) {
      setOutput(`Error: ${error.message}`);
    } finally {
      setIsRunning(false);
    }
  };
  


  return (
    <div className="code-editor-container">
      <div className="editor-header">
        <h3>{selectedSnippet}</h3>
        <button onClick={runCode} disabled={isRunning}>
          {isRunning ? 'Running...' : 'Run Code'}
        </button>
      </div>
      
      <div className="editor-layout">
        <Editor
          height="400px"
          language={snippet.language}
          value={code}
          theme="vs-dark"
          onChange={handleEditorChange}
          options={{
            fontSize: 14,
            minimap: { enabled: false },
            scrollBeyondLastLine: false,
            wordWrap: 'on',
            automaticLayout: true,
          }}
        />
        
        <div className="output-panel">
          <h4>Output:</h4>
          <pre className="output">{output}</pre>
        </div>
      </div>
    </div>
  );

  // return (
  //   <Editor
  //     height="400px"
  //     language={snippet.language}
  //     value={snippet.code}
  //     theme="vs-light"
  //     options={{
  //       readOnly: true, // Start with read-only
  //       minimap: { enabled: false },
  //       scrollBeyondLastLine: false,
  //     }}
  //   />
  // );
  // const { code, language } = snippet;

  // return (
  //   <div>
  //     <SyntaxHighlighter language={language} style={solarizedlight}>
  //       {code}
  //     </SyntaxHighlighter>
  //   </div>
  // );
}

export default CodePane;
