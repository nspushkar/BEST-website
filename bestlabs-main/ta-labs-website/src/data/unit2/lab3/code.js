// Define all your code snippets here with languages
const codeSnippets = {
    full: {
        code: `# Full Genetic Algorithm Code for Lab 3

# Step 1: Import Libraries and Define the Fitness Function
import numpy as np
import random

# Fitness function example (minimizing the sum of squares)
def fitness(individual):
    return sum(x**2 for x in individual)

# Step 2: Initialize Population
def generate_population(size, length, min_value, max_value):
    population = []
    for _ in range(size):
        individual = [random.uniform(min_value, max_value) for _ in range(length)]
        population.append(individual)
    return population

population = generate_population(10, 5, -10, 10)

# Step 3: Selection (Roulette Wheel Selection)
def roulette_wheel_selection(population, fitness_fn):
    max_fitness = sum(fitness_fn(ind) for ind in population)
    pick = random.uniform(0, max_fitness)
    current = 0
    for individual in population:
        current += fitness_fn(individual)
        if current > pick:
            return individual

selected = roulette_wheel_selection(population, fitness)

# Step 4: Crossover and Mutation
def crossover(parent1, parent2, crossover_rate=0.8):
    if random.random() < crossover_rate:
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    return parent1, parent2

def mutate(individual, mutation_rate=0.01):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.uniform(-10, 10)
    return individual

child1, child2 = crossover(population[0], population[1])
child1 = mutate(child1)
child2 = mutate(child2)

# Step 5: Evaluation and Replacement
def replace_population(population, new_population, fitness_fn):
    combined = population + new_population
    combined.sort(key=fitness_fn)
    return combined[:len(population)]

new_population = [mutate(crossover(*random.sample(population, 2))[0]) for _ in range(len(population))]
population = replace_population(population, new_population, fitness)
`,
        language: 'python'
    },

    fitness_function: {
        code: `# Step 1: Define the Fitness Function
def fitness(individual):
    return sum(x**2 for x in individual)

individual = [random.uniform(-10, 10) for _ in range(5)]
print(f"Fitness of the individual: {fitness(individual)}")
`,
        language: 'python'
    },

    population_generation: {
        code: `# Step 2: Initialize Population
def generate_population(size, length, min_value, max_value):
    population = []
    for _ in range(size):
        individual = [random.uniform(min_value, max_value) for _ in range(length)]
        population.append(individual)
    return population

population = generate_population(10, 5, -10, 10)
print("Initial Population:", population)
`,
        language: 'python'
    },

    selection: {
        code: `# Step 3: Selection using Roulette Wheel
def roulette_wheel_selection(population, fitness_fn):
    max_fitness = sum(fitness_fn(ind) for ind in population)
    pick = random.uniform(0, max_fitness)
    current = 0
    for individual in population:
        current += fitness_fn(individual)
        if current > pick:
            return individual

selected = roulette_wheel_selection(population, fitness)
print("Selected Individual:", selected)
`,
        language: 'python'
    },

    mutation_crossover: {
        code: `# Step 4: Crossover and Mutation
def crossover(parent1, parent2, crossover_rate=0.8):
    if random.random() < crossover_rate:
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    return parent1, parent2

def mutate(individual, mutation_rate=0.01):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.uniform(-10, 10)
    return individual

child1, child2 = crossover(population[0], population[1])
child1 = mutate(child1)
child2 = mutate(child2)
`,
        language: 'python'
    },

    evaluation: {
        code: `# Step 5: Evaluation and Replacement
def replace_population(population, new_population, fitness_fn):
    combined = population + new_population
    combined.sort(key=fitness_fn)
    return combined[:len(population)]

new_population = [mutate(crossover(*random.sample(population, 2))[0]) for _ in range(len(population))]
population = replace_population(population, new_population, fitness)
print("New Population:", population)
`,
        language: 'python'
    }
};

export default codeSnippets;
