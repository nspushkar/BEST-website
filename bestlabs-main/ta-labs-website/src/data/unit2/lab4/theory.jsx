import React from 'react';
import bajajService from './imgs/bajajservice.avif'

const Lab4 = ({ onLinkClick }) => (
  <div className="p-6">
    <h1 className="text-2xl font-bold mb-4">
      <a href="#full" className="link-primary" onClick={() => onLinkClick('full')}>
        Multi-Level Diagnosis Theory Explained
      </a>
    </h1>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#problem_domain" className="link-primary" onClick={() => onLinkClick('facts')}>
        Step 1: Defining the Problem Domain
      </a>
    </h2>
    <p className="mb-4">
      In this expert system, we focus on diagnosing issues related to motorcycle engines. The expert system will cover
      engine problems such as overheating, stalling, misfiring, and strange noises. Symptoms provided by the user
      will be used to diagnose potential faults and explore their causes at multiple levels. You input the symptom, and
      the system will provide a diagnosis, its root cause, and the recommended fix.
    </p>

    {/* Image Section */}
    <div className="mt-6">
      <img src={bajajService} alt="Bajaj Motorcycle Service" className="w-4/5 mx-auto my-4 p-2" />
      <p className="text-center text-gray-700">Bajaj Motorcycle Service Diagnostic Overview</p>
    </div>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#rules" className="link-primary" onClick={() => onLinkClick('rules')}>
        Step 2: Writing Rules for Deduction
      </a>
    </h2>
    <p className="mb-4">
      The core of this expert system involves defining rules for diagnosing engine faults based on observed symptoms.
      Each primary diagnosis is followed by secondary-level reasoning that explores the possible causes of the issue.
      For example, engine overheating can be caused by a coolant system failure, which itself can result from a blocked
      radiator or a broken fan.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#multi_level_diagnosis" className="link-primary" onClick={() => onLinkClick('multilvldiagnosis')}>
        Step 3: Multi-Level Diagnosis
      </a>
    </h2>
    <p className="mb-4">
      The multi-level diagnosis approach allows the system to deduce not only the primary fault (e.g., engine overheating)
      but also explore why that fault occurred (e.g., broken fan or blocked radiator). It then suggests the best next steps
      to resolve the issue, such as checking the coolant system or replacing a faulty part. You can query for both the
      diagnosis and the recommended action in one step.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#recommendations" className="link-primary" onClick={() => onLinkClick('queries')}>
        Step 4: Providing Recommendations
      </a>
    </h2>
    <p className="mb-4">
      Based on the identified cause, the system provides recommendations for corrective actions. For example, if a coolant
      system failure is detected due to a broken coolant fan, the system will recommend inspecting the fan and possibly
      replacing it. These recommendations provide actionable insights for resolving the diagnosed issues.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#further_development" className="link-primary" onClick={() => onLinkClick('full')}>
        Step 5: Further Development
      </a>
    </h2>
    <p className="mb-4">
      This multi-level diagnosis system can be expanded by including more symptoms, deeper diagnostic rules, and
      additional subsystems. In the future, we can integrate other mechanical parts such as the motorcycle transmission
      or braking system into the model.
    </p>
  </div>
);

export default Lab4;
