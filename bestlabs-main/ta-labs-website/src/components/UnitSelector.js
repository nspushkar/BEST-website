import React from 'react';

function UnitSelector({ units, selectedUnit, onSelectUnit }) {
  return (
    <div className="container mx-auto mb-8 py-8 bg-white">
      <div className="flex flex-row justify-center space-x-6">
        {units.map((unit, index) => (
          <button
            key={unit.id}
            className={`px-16 py-12 rounded border ${
              selectedUnit && selectedUnit.id === unit.id
                ? 'bg-[#FF8300] text-[#FDF6E3]'
                : 'bg-[#FDF6E3] text-[#FF8300] hover:[#d0e0f0'
            } transition-colors duration-300 text-5xl`}
            onClick={() => onSelectUnit(unit.id)}
          >
            Unit {index + 1}
          </button>
        ))}
      </div>
      {selectedUnit && (
        <div className="mt-8 text-center">
          <h2 className="text-3xl font-bold mb-2">{selectedUnit.name}</h2>
        </div>
      )}
    </div>
  );
}

export default UnitSelector;
