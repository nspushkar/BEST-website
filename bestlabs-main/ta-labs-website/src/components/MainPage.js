import React, { useState, useRef } from 'react';
import Hero from './Hero';
import UnitSelector from './UnitSelector';
import LabList from './LabList';
import units from '../data/units'; // Ensure correct import path for units data

function MainPage() {
  const [selectedUnit, setSelectedUnit] = useState(null);

  const handleSelectUnit = (unitId) => {
    const unit = units.find(unit => unit.id === unitId);
    setSelectedUnit(unit);
  };

  return (
    <>
      <Hero />
      <UnitSelector
        units={units}
        selectedUnit={selectedUnit}
        onSelectUnit={handleSelectUnit}
      />
      <LabList unit={selectedUnit} labs={selectedUnit ? selectedUnit.labs : []} />
    </>
  );
}

export default MainPage;
