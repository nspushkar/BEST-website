import React, { useEffect, Suspense } from 'react';
import Lab1 from './lab1content.jsx';  // Import Lab1 component

function LabDetail({ lab, unit, onBack }) {

  useEffect(() => {
    // Scroll to top on page load
    window.scrollTo(0, 0);
  }, [lab, unit]);

  return (
    <div className="bg-[#f0f1f2] -mx-4">
      <div className="px-0 py-0 min-h-screen">
        {/* Header Section */}
        <div className="bg-[#d0e0f0] text-[#113b7d] py-12 px-4">
          <div className="container mx-auto flex justify-between items-center">
            <div>
              <h1 className="text-4xl font-bold mb-2">Lab 1: AI Dev Environment</h1>
              <p className="text-xl font-bold mb-4">Setting up an AI Development Environment</p>
            </div>
            <div className="flex flex-1 justify-end"> {/* Add flex-1 and justify-end to right-align */}
              <button 
                onClick={onBack} 
                className="bg-[#113b7d] text-white py-2 px-6 rounded-lg shadow-md hover:bg-[#0f2e5b] transition-colors duration-300"
              >
                Back to labs
              </button>
            </div>
          </div>
        </div>

        <div className="container mx-auto py-8 flex space-x-8 bg-[#f0f1f2]">
          {/* Theory Pane */}
          <div className="bg-[#FDF6E3] p-6 rounded-lg shadow-md">
            <h2 className="text-2xl font-bold mb-4">Theory</h2>
            <div className="text-gray-700 scroll-pane">
              <Suspense fallback={<div>Loading theory...</div>}>
                {Lab1 ? <Lab1 /> : <div>Loading theory content...</div>}
              </Suspense>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default LabDetail;
