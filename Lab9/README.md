# Lab 3: Genome & Evolutionary Strategy

## Overview

This code implements an evolutionary algorithm to solve an optimization problem. The goal is to minimize fitness calls, avoiding unnecessary extra calls that don't contribute significantly to the search for the best fitness. The genome represents the genetic makeup of an individual in the population, consisting of a sequence of binary values (0 or 1).

## Evolutionary Process

The population evolves through successive generations. Each generation involves evaluating individuals based on a fitness function. The best individuals, known as elite, are preserved in the next generation. The remaining individuals are selected through random tournaments, undergo crossover and mutation, introducing genetic variations to discover potentially better solutions.

The algorithm returns the individual with the highest fitness from the final population as the optimal or approximate solution to the problem.

## Key Components

### Parent Selection

Parent selection involves elite selection, preserving the best individuals, and random choices among elites, introducing randomness for genetic diversity.

### Crossover

The crossover function combines information from two parents to generate a child. The crossover point, randomly chosen between the second and the last element of the genome, maintains genetic diversity.

### Mutation

The mutate function applies mutation to an individual, flipping each bit with a certain probability.

## Simulation

The simulation tests various configurations of parameters (POPULATION_SIZE, MUTATION_RATE, GENERATIONS, ELITISM_PERCENTAGE) to find the optimal setup yielding the best solution for each problem.

## Results

### Best Solution Overall:

- Population size: 1000
- Generations: 100
- Mutation rate: 0.01
- Elitism percentage: 0.3
- Fitness: 75.00%
- Number of fitness calls: 75253

## Conclusion

The evolutionary algorithm demonstrates effective optimization, providing insights into parameter configurations for improved performance.

