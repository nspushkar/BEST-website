import React, { useState } from 'react';
import Header from './components/Header';
import Hero from './components/Hero';
import UnitSelector from './components/UnitSelector';
import LabList from './components/LabList';
import LabDetail from './components/LabDetail'; // Component for individual lab pages
import units from './data/units';
import Lab1 from './data/unit1/lab1/lab1'
import Footer from './components/Footer'

function App() {
  const [selectedUnit, setSelectedUnit] = useState(null);
  const [selectedLab, setSelectedLab] = useState(null);

  const handleSelectUnit = (unitId) => {
    const unit = units.find(unit => unit.id === unitId);
    setSelectedUnit(unit);
    setSelectedLab(null); // Reset selected lab when unit changes
  };

  const handleSelectLab = (labId) => {
    const lab = selectedUnit.labs.find(lab => lab.id === labId);
    setSelectedLab(lab);
  };

  const handleLabsClick = () => {
    setSelectedLab(null); // Go back to the main page when "Labs" is clicked
  };

  return (
    <div className="App">
      <Header onLabsClick={handleLabsClick} />
      {!selectedLab ? (
        <>
          <Hero />
          <UnitSelector units={units} selectedUnit={selectedUnit} onSelectUnit={handleSelectUnit} />
          <LabList labs={selectedUnit ? selectedUnit.labs : []} onSelectLab={handleSelectLab} />
        </>
      ) : (
        selectedUnit.id === 1 && selectedLab.id === 1 ? (
          <Lab1 lab={selectedLab} unit={selectedUnit} onBack={() => setSelectedLab(null)} />
        ) : (
          <LabDetail lab={selectedLab} unit={selectedUnit} onBack={() => setSelectedLab(null)} />
        )
      )}
      <Footer/>
    </div>
  );
}

export default App;
