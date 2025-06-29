import React from 'react';
import { TypeAnimation } from 'react-type-animation';

function Hero() {
    return (
        <section className="min-h-[60vh] flex items-center justify-center bg-[#d0e0f0] text-primary py-12">
            <div className="text-start max-w-4xl mx-auto">
                <TypeAnimation
                    sequence={[
                        'Artificial Intelligence Lab Portal.',
                        1000, // Delay before looping
                    ]}
                    speed={50} // Speed of typing
                    wrapper="h1"
                    repeat={1} // Loop count (1 for no loop, or Infinity for continuous looping)
                    className="text-5xl md:text-7xl lg:text-9xl font-bold leading-tight"
                />
            </div>
        </section>
    );
}

export default Hero;
