import React from 'react';

function LabList({ labs, selectedUnit, onSelectLab }) {
    if (!labs.length) {
        return (
            <div>
                <p className="text-center text-gray-500">Please select a unit.</p>
                <br />
            </div>
        );
    }

    return (
        <div className="bg-[#f0f1f2] -mx-4"> {/* Apply background color with negative margin */}
            <section id="labs" className="container mx-auto py-4 px-4"> {/* Keep original padding */}
                {selectedUnit && (
                    <div className="mb-12 text-center">
                        <h2 className="text-4xl font-bold mb-4">Unit {selectedUnit.id}</h2>
                        <p className="text-xl text-gray-700 mb-8">{selectedUnit.description}</p>
                        <h3 className="text-3xl font-bold mb-8">Available Labs</h3>
                    </div>
                )}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8">
                    {labs.map(lab => (
                        <div
                            key={lab.id}
                            className="bg-white p-6 rounded-lg shadow-lg hover:shadow-xl transition-shadow duration-300 flex flex-col justify-between"
                        >
                            <div>
                                <h3 className="text-2xl font-bold mb-3">{lab.name}</h3>
                                <p className="text-gray-700 mb-4">{lab.description}</p>
                            </div>
                            <div className="mt-auto"> {/* Ensures the button stays at the bottom */}
                                <button
                                    onClick={() => onSelectLab(lab.id)}
                                    className="text-blue-600 hover:text-blue-800 transition-colors duration-300 font-semibold"
                                >
                                    Learn More
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            </section>
        </div>
    );
}

export default LabList;
