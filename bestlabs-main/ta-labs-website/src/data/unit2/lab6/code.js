// Define all your code snippets here with languages
const codeSnippets = {
    full: {
        code: `% Complete Code for Vehicle Model Identification

% Models
model(sedan).
model(suv).
model(truck).
model(coupe).
model(hatchback).
model(convertible).
model(minivan).
model(crossover).
model(pickup).
model(sports_car).

% Characteristics
characteristic(four_doors).
characteristic(two_doors).
characteristic(all_wheel_drive).
characteristic(front_wheel_drive).
characteristic(rear_wheel_drive).
characteristic(higher_ground_clearance).
characteristic(compact_size).
characteristic(large_cargo_space).
characteristic(removable_roof).
characteristic(third_row_seating).
characteristic(towing_capacity).
characteristic(fuel_efficient).
characteristic(luxury_interior).
characteristic(high_performance).

% Rules for identifying each model
rule(sedan, [four_doors, front_wheel_drive, fuel_efficient, luxury_interior]).
rule(suv, [four_doors, all_wheel_drive, higher_ground_clearance, large_cargo_space, third_row_seating]).
rule(truck, [four_doors, rear_wheel_drive, large_cargo_space, towing_capacity]).
rule(coupe, [two_doors, rear_wheel_drive, high_performance, luxury_interior]).
rule(hatchback, [four_doors, compact_size, fuel_efficient, front_wheel_drive]).
rule(convertible, [two_doors, removable_roof, high_performance, luxury_interior]).
rule(minivan, [four_doors, front_wheel_drive, large_cargo_space, third_row_seating]).
rule(crossover, [four_doors, all_wheel_drive, compact_size, fuel_efficient]).
rule(pickup, [two_doors, rear_wheel_drive, large_cargo_space, towing_capacity]).
rule(sports_car, [two_doors, high_performance, luxury_interior]).

% Decision Support System: Identify model based on characteristics
identify_model(Model) :-
    model(Model),
    rule(Model, Characteristics),
    ask_characteristics(Characteristics).

% Function to prompt user for characteristic confirmation
ask_characteristics([]).
ask_characteristics([Characteristic | Rest]) :-
    format('Does the model have the following characteristic: ~w? (yes/no) ', [Characteristic]),
    read(Response),
    (Response == yes -> characteristic(Characteristic) ; fail),
    ask_characteristics(Rest).`,
        language: 'prolog'
    },

    models: {
        code: `% Vehicle Models
model(sedan).
model(suv).
model(truck).
model(coupe).
model(hatchback).
model(convertible).
model(minivan).
model(crossover).
model(pickup).
model(sports_car).`,
        language: 'prolog'
    },

    characteristics: {
        code: `% Vehicle Characteristics
characteristic(four_doors).
characteristic(two_doors).
characteristic(all_wheel_drive).
characteristic(front_wheel_drive).
characteristic(rear_wheel_drive).
characteristic(higher_ground_clearance).
characteristic(compact_size).
characteristic(large_cargo_space).
characteristic(removable_roof).
characteristic(third_row_seating).
characteristic(towing_capacity).
characteristic(fuel_efficient).
characteristic(luxury_interior).
characteristic(high_performance).`,
        language: 'prolog'
    },

    rules: {
        code: `% Rules for Identifying Vehicle Models
rule(sedan, [four_doors, front_wheel_drive, fuel_efficient, luxury_interior]).
rule(suv, [four_doors, all_wheel_drive, higher_ground_clearance, large_cargo_space, third_row_seating]).
rule(truck, [four_doors, rear_wheel_drive, large_cargo_space, towing_capacity]).
rule(coupe, [two_doors, rear_wheel_drive, high_performance, luxury_interior]).
rule(hatchback, [four_doors, compact_size, fuel_efficient, front_wheel_drive]).
rule(convertible, [two_doors, removable_roof, high_performance, luxury_interior]).
rule(minivan, [four_doors, front_wheel_drive, large_cargo_space, third_row_seating]).
rule(crossover, [four_doors, all_wheel_drive, compact_size, fuel_efficient]).
rule(pickup, [two_doors, rear_wheel_drive, large_cargo_space, towing_capacity]).
rule(sports_car, [two_doors, high_performance, luxury_interior]).`,
        language: 'prolog'
    },

    decision_support: {
        code: `% Decision Support System to Identify Vehicle Model
identify_model(Model) :-
    model(Model),
    rule(Model, Characteristics),
    ask_characteristics(Characteristics).

ask_characteristics([]).
ask_characteristics([Characteristic | Rest]) :-
    format('Does the model have the following characteristic: ~w? (yes/no) ', [Characteristic]),
    read(Response),
    (Response == yes -> characteristic(Characteristic) ; fail),
    ask_characteristics(Rest).`,
        language: 'prolog'
    }
};

export default codeSnippets;
