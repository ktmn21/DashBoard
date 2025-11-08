import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import random

class GeneticAlgorithm:
    def __init__(self, population_size=50, mutation_rate=0.1, crossover_rate=0.8, generations=50):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.fitness_history = []
        self.best_history = []
    
    def fitness_function(self, individual):
        """Example: Maximize sum of squares (simple optimization)"""
        return sum(x**2 for x in individual)
    
    def create_individual(self, size=10, bounds=(-10, 10)):
        """Create a random individual"""
        return [random.uniform(bounds[0], bounds[1]) for _ in range(size)]
    
    def create_population(self, size, individual_size=10, bounds=(-10, 10)):
        """Create initial population"""
        return [self.create_individual(individual_size, bounds) for _ in range(size)]
    
    def select_parents(self, population, fitness_scores):
        """Tournament selection"""
        tournament_size = 3
        parents = []
        for _ in range(2):
            tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
            winner = max(tournament, key=lambda x: x[1])
            parents.append(winner[0])
        return parents
    
    def crossover(self, parent1, parent2):
        """Single-point crossover"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    
    def mutate(self, individual, bounds=(-10, 10)):
        """Gaussian mutation"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutated[i] += random.gauss(0, 1)
                mutated[i] = max(bounds[0], min(bounds[1], mutated[i]))
        return mutated
    
    def evolve(self, individual_size=10, bounds=(-10, 10)):
        """Run genetic algorithm"""
        population = self.create_population(self.population_size, individual_size, bounds)
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [self.fitness_function(ind) for ind in population]
            avg_fitness = np.mean(fitness_scores)
            best_fitness = max(fitness_scores)
            best_individual = population[np.argmax(fitness_scores)]
            
            self.fitness_history.append(avg_fitness)
            self.best_history.append(best_fitness)
            
            # Create new population
            new_population = []
            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents(population, fitness_scores)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1, bounds)
                child2 = self.mutate(child2, bounds)
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        # Final evaluation
        final_fitness = [self.fitness_function(ind) for ind in population]
        best_idx = np.argmax(final_fitness)
        
        return population[best_idx], max(final_fitness)

def show_genetic():
    st.header("Genetic Algorithm")
    st.markdown("""
    Genetic Algorithms are optimization techniques inspired by natural selection.
    They evolve a population of solutions over generations using selection, 
    crossover, and mutation.
    """)
    
    st.subheader("Algorithm Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        population_size = st.slider("Population Size", 10, 200, 50)
        generations = st.slider("Number of Generations", 10, 200, 50)
    
    with col2:
        mutation_rate = st.slider("Mutation Rate", 0.0, 1.0, 0.1)
        crossover_rate = st.slider("Crossover Rate", 0.0, 1.0, 0.8)
    
    st.write("**Fitness Function:** Maximize sum of squares (Σx²)")
    
    if st.button("Run Algorithm", type="primary"):
        ga = GeneticAlgorithm(
            population_size=population_size,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            generations=generations
        )
        
        with st.spinner("Evolving population..."):
            best_solution, best_fitness = ga.evolve()
        
        st.success(f"Best Fitness: {best_fitness:.2f}")
        st.write(f"Best Solution (first 10 values): {[f'{x:.2f}' for x in best_solution[:10]]}")
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Fitness evolution
        generations_list = list(range(len(ga.fitness_history)))
        ax1.plot(generations_list, ga.fitness_history, 'b-', label='Average Fitness', alpha=0.7)
        ax1.plot(generations_list, ga.best_history, 'r-', label='Best Fitness', linewidth=2)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Fitness Evolution Over Generations')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Population diversity (last generation)
        last_gen_fitness = [ga.fitness_function(ind) for ind in 
                           ga.create_population(ga.population_size)]
        ax2.hist(last_gen_fitness, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.axvline(best_fitness, color='red', linestyle='--', linewidth=2, label='Best')
        ax2.set_xlabel('Fitness')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Fitness Distribution (Last Generation)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show statistics
        st.subheader("Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Initial Avg Fitness", f"{ga.fitness_history[0]:.2f}")
        with col2:
            st.metric("Final Avg Fitness", f"{ga.fitness_history[-1]:.2f}")
        with col3:
            improvement = ((ga.best_history[-1] - ga.best_history[0]) / ga.best_history[0]) * 100
            st.metric("Improvement", f"{improvement:.1f}%")

