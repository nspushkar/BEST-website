% Multi-Level Motorcycle Engine Diagnosis

% Facts: Symptoms related to motorcycle engine issues
symptom(engine_overheating).
symptom(strange_engine_noise).
symptom(engine_stalling).
symptom(poor_fuel_efficiency).
symptom(engine_hard_to_start).
symptom(smoke_from_exhaust).

% Primary Diagnosis: Identifying the main issue based on symptoms
primary_problem(coolant_system_failure) :- symptom(engine_overheating).
primary_problem(fuel_system_issue) :- symptom(engine_stalling), symptom(poor_fuel_efficiency).
primary_problem(ignition_system_issue) :- symptom(engine_hard_to_start).
primary_problem(exhaust_issue) :- symptom(smoke_from_exhaust), symptom(strange_engine_noise).

% Secondary Diagnosis: Investigating causes of the primary problem

% Cooling System Failure Causes
cause(broken_coolant_fan) :- primary_problem(coolant_system_failure), symptom(engine_overheating).
cause(blocked_radiator) :- primary_problem(coolant_system_failure), + symptom(smoke_from_exhaust).

% Fuel System Issue Causes
cause(clogged_fuel_filter) :- primary_problem(fuel_system_issue), symptom(poor_fuel_efficiency).
cause(fuel_pump_failure) :- primary_problem(fuel_system_issue), + symptom(poor_fuel_efficiency).

% Ignition System Issue Causes
cause(faulty_spark_plugs) :- primary_problem(ignition_system_issue).
cause(ignition_coil_failure) :- primary_problem(ignition_system_issue), + symptom(smoke_from_exhaust).

% Exhaust System Issue Causes
cause(broken_exhaust_pipe) :- primary_problem(exhaust_issue), symptom(smoke_from_exhaust).
cause(piston_ring_failure) :- primary_problem(exhaust_issue), symptom(strange_engine_noise).

% Recommendations: Actions based on causes
recommendation(check_coolant_fan) :- cause(broken_coolant_fan).
recommendation(clear_radiator) :- cause(blocked_radiator).
recommendation(replace_fuel_filter) :- cause(clogged_fuel_filter).
recommendation(check_fuel_pump) :- cause(fuel_pump_failure).
recommendation(replace_spark_plugs) :- cause(faulty_spark_plugs).
recommendation(check_ignition_coil) :- cause(ignition_coil_failure).
recommendation(replace_exhaust_pipe) :- cause(broken_exhaust_pipe).
recommendation(check_piston_rings) :- cause(piston_ring_failure).

% Full Diagnosis: Input a symptom and return the full diagnosis with recommendations
full_diagnosis(Symptom, PrimaryDiagnosis, Cause, Recommendation) :-
    symptom(Symptom),
    primary_problem(PrimaryDiagnosis),
    cause(Cause),
    recommendation(Recommendation).

% Example Queries:

% For overheating engine:
% ?- full_diagnosis(engine_overheating, PrimaryDiagnosis, Cause, Recommendation).

% For stalling engine with poor fuel efficiency:
% ?- full_diagnosis(engine_stalling, PrimaryDiagnosis, Cause, Recommendation).