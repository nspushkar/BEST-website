import React, { useEffect, useState, Suspense } from 'react';

function LabDetail({ lab, unit, onBack }) {
  const [Theory, setTheory] = useState(null);
  const [CodePane, setCodePane] = useState(null);
  const [selectedSnippet, setSelectedSnippet] = useState('full'); // Default to 'full'
  const [runnableCodeLink, setRunnableCodeLink] = useState('');

  useEffect(() => {
    // Dynamically import the Theory component
    import(`../data/unit${unit.id}/lab${lab.id}/theory.jsx`)
      .then(module => setTheory(() => module.default))
      .catch(err => console.error('Failed to load theory content', err));

    // Dynamically import the CodePane component
    import(`../data/unit${unit.id}/lab${lab.id}/codepane.js`)
      .then(module => setCodePane(() => module.default))
      .catch(err => console.error('Failed to load code pane component', err));

    // Load the runnable code links
    fetch('/data/runnableCodeLinks.json')
      .then(response => response.json())
      .then(data => {
        setRunnableCodeLink(data[`unit${unit.id}`][`lab${lab.id}`]);
      })
      .catch(err => console.error('Failed to load runnable code links', err));

    // Scroll to top on page load
    window.scrollTo(0, 0);
  }, [lab, unit]);

  const handleLinkClick = (snippet) => {
    setSelectedSnippet(snippet);
  };

  const handleViewRunnableCode = () => {
    if (runnableCodeLink) {
      window.open(runnableCodeLink, '_blank');
    } else {
      alert('Runnable code link not available.');
    }
  };

  return (
    <div className="bg-[#f0f1f2] -mx-4">
      <div className="px-0 py-0 min-h-screen">
        {/* Header Section */}
        <div className="bg-[#d0e0f0] text-[#113b7d] py-12 px-4">
          <div className="container mx-auto flex justify-between items-center">
            <div>
              <h1 className="text-4xl font-bold mb-2">{lab.name}</h1>
              <p className="text-xl font-bold mb-4">{unit.name}</p>
            </div>
            <div className="flex space-x-4">
              <button 
                onClick={handleViewRunnableCode} 
                className="bg-[#113b7d] text-white py-2 px-6 rounded-lg shadow-md hover:bg-[#0f2e5b] transition-colors duration-300"
              >
                View Runnable Code
              </button>
              <button 
                onClick={onBack} 
                className="bg-[#113b7d] text-white py-2 px-6 rounded-lg shadow-md hover:bg-[#0f2e5b] transition-colors duration-300"
              >
                Back to Labs
              </button>
            </div>
          </div>
        </div>

        {/* Two-Pane Layout */}
        <div className="container mx-auto py-8 flex space-x-8 bg-[#f0f1f2]">
          {/* Theory Pane */}
          <div className="w-1/2 bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-2xl font-bold mb-4">Theory</h2>
            <div className="text-gray-700 scroll-pane">
              <Suspense fallback={<div>Loading theory...</div>}>
                {Theory ? <Theory onLinkClick={handleLinkClick} /> : <div>Loading theory content...</div>}
              </Suspense>
            </div>
          </div>

          {/* Code Pane */}
          <div className="w-1/2 bg-[#FDF6E3] p-6 rounded-lg shadow-md">
            <h2 className="text-2xl font-bold mb-4">Code</h2>
            <div className="text-gray-700 scroll-pane">
              <Suspense fallback={<div>Loading code pane...</div>}>
                {CodePane ? <CodePane selectedSnippet={selectedSnippet} /> : <div>Loading code pane component...</div>}
              </Suspense>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default LabDetail;
