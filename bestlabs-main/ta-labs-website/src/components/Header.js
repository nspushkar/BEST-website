import React from 'react';
import pesLogo from '../logos/peslogo.png';
import bajajLogo from '../logos/Bajaj-Motorcycles-Logo.png';

function Header({ onLabsClick }) {
  return (
    <header className="relative bg-white shadow-md py-4">
      <div className="container mx-auto flex items-center justify-between relative">
        {/* University Logo */}
        <img src={pesLogo} alt="University Logo" className="h-12" />

        {/* Centered Text */}
        <div className="absolute inset-x-0 top-1/2 transform -translate-y-1/2 flex justify-center">
          <span className="text-xl font-semibold text-[#002d74]">Bajaj Engineering Skills Training</span>
        </div>

        {/* Bajaj Logo */}
        <img src={bajajLogo} alt="Bajaj Logo" className="h-12" />
      </div>
    </header>
  );
}

export default Header;
