import React from 'react';
import mutationcrossover from './imgs/mutationcrossover.png'
import cgp from './imgs/cgp.png'

const Theory = ({ onLinkClick }) => (
  <div className="p-6">
    <h1 className="text-2xl font-bold mb-4">
      <a href="#genetic_algorithm" className="link-primary" onClick={() => onLinkClick('full')}>
        Unit 2 Lab 3: Genetic Algorithm Theory
      </a>
    </h1>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#basic_concepts" className="link-primary" onClick={() => onLinkClick('full')}>
        Basic Concepts of Genetic Algorithms
      </a>
    </h2>
    <p className="mb-4">
      Genetic Algorithms (GA) are search heuristics inspired by Charles Darwin’s theory of natural evolution. The process reflects how living organisms evolve over time through selection, crossover, and mutation. In a GA, a population of candidate solutions evolves toward better solutions. Each solution has a set of properties (its “chromosome”) that can be mutated and altered. GAs are used to solve both optimization and search problems.
    </p>

    <img src={cgp} alt="Chromosome, Gene, Population" className="w-4/5 mx-auto my-4 p-2" />

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#fitness_function" className="link-primary" onClick={() => onLinkClick('fitness_function')}>
        Fitness Function
      </a>
    </h2>
    <p className="mb-4">
      The fitness function evaluates how close a given solution is to the optimal solution of the problem. The better the fitness score, the more likely that solution is to be selected for reproduction. 
      The goal is to evolve a population over generations, continually improving the fitness of the population by combining the best solutions.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#selection" className="link-primary" onClick={() => onLinkClick('selection')}>
        Selection
      </a>
    </h2>
    <p className="mb-4">
      Selection is the process of choosing individuals from a population to produce the next generation. Individuals with higher fitness scores are more likely to be selected. Techniques like roulette wheel selection are commonly used, where solutions are chosen with a probability proportional to their fitness scores.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#crossover" className="link-primary" onClick={() => onLinkClick('mutation_crossover')}>
        Crossover
      </a>
    </h2>
    <p className="mb-4">
      Crossover combines two parents to produce offspring, passing on a portion of each parent’s genetic information. Crossover allows the algorithm to explore new areas of the solution space. This process mimics biological reproduction, where the genetic material from two parents is recombined to create a child.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#mutation" className="link-primary" onClick={() => onLinkClick('mutation_crossover')}>
        Mutation
      </a>
    </h2>
    <p className="mb-4">
      Mutation introduces randomness into the population to maintain genetic diversity. During mutation, one or more genes in a chromosome are randomly altered. This process helps prevent the algorithm from converging too quickly to a local optimum and keeps the population diverse enough to explore new solutions.
    </p>

    <img src={mutationcrossover} alt="Linear Regression Best fit line" className="w-4/5 mx-auto my-4 p-2" />

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#termination" className="link-primary" onClick={() => onLinkClick('evaluation')}>
        Termination
      </a>
    </h2>
    <p className="mb-4">
      The GA process continues for a fixed number of generations or until a satisfactory fitness level is achieved. The final output of the algorithm is the solution with the highest fitness score after the termination criteria are met.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#step1" className="link-primary" onClick={() => onLinkClick('population_generation')}>
        Step 1: Population Initialization
      </a>
    </h2>
    <p className="mb-4">
      The notebook begins by defining the initial population. Each individual in the population is a potential solution represented by a list of numbers (chromosome). The initial population is generated randomly within a given range for each gene.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#step2" className="link-primary" onClick={() => onLinkClick('fitness_function')}>
        Step 2: Fitness Function
      </a>
    </h2>
    <p className="mb-4">
      The fitness function is then defined, which evaluates how well a given solution solves the problem. The fitness score guides the selection process for the next generation.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#step3" className="link-primary" onClick={() => onLinkClick('selection')}>
        Step 3: Selection
      </a>
    </h2>
    <p className="mb-4">
      Selection involves choosing individuals from the current population to serve as parents for the next generation. The notebook uses a roulette wheel selection method, where individuals with better fitness are more likely to be selected.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#step4" className="link-primary" onClick={() => onLinkClick('mutation_crossover')}>
        Step 4: Crossover and Mutation
      </a>
    </h2>
    <p className="mb-4">
      After selection, crossover is performed to generate new offspring by combining genes from the parent solutions. Afterward, mutation introduces slight random changes to some of the genes to explore new solutions and maintain diversity in the population.
    </p>

    <h2 className="text-xl font-semibold mt-6 mb-2">
      <a href="#step5" className="link-primary" onClick={() => onLinkClick('evaluation')}>
        Step 5: Evaluation
      </a>
    </h2>
    <p className="mb-4">
      Finally, the fitness of the new population is evaluated, and the best individuals are retained for the next generation. This process repeats until a stopping condition (e.g., a maximum number of generations) is reached.
    </p>
  </div>
);

export default Theory;
