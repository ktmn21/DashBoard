import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd

class GeneticAlgorithm:
    def __init__(self, population_size=50, mutation_rate=0.1, crossover_rate=0.8, generations=50):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.fitness_history = []
        self.best_history = []
        self.phases_data = []  # Store data for each phase
    
    def fitness_function(self, individual):
        """Example: Maximize sum of squares (simple optimization)"""
        return sum(x**2 for x in individual)
    
    def create_individual(self, size=10, bounds=(-10, 10)):
        """Create a random individual (chromosome)"""
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
        mutation_indices = []
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutated[i] += random.gauss(0, 1)
                mutated[i] = max(bounds[0], min(bounds[1], mutated[i]))
                mutation_indices.append(i)
        return mutated, mutation_indices

def show_genetic():
    st.header("Genetic Algorithm")
    st.markdown("""
    Genetic Algorithms are optimization techniques inspired by natural selection.
    They evolve a population of solutions over generations using five key phases:
    1. **Initial Population**: Random chromosomes
    2. **Fitness Function**: Evaluate each individual
    3. **Selection**: Choose parents for reproduction
    4. **Crossover**: Create offspring by combining parents
    5. **Mutation**: Introduce random changes
    """)
    
    # Initialize session state
    if 'ga_phase' not in st.session_state:
        st.session_state.ga_phase = 0
    if 'ga_generation' not in st.session_state:
        st.session_state.ga_generation = 0
    if 'ga_population' not in st.session_state:
        st.session_state.ga_population = None
    if 'ga_fitness' not in st.session_state:
        st.session_state.ga_fitness = None
    if 'ga_selected' not in st.session_state:
        st.session_state.ga_selected = []
    if 'ga_offspring' not in st.session_state:
        st.session_state.ga_offspring = []
    if 'ga_mutated' not in st.session_state:
        st.session_state.ga_mutated = []
    if 'ga_history' not in st.session_state:
        st.session_state.ga_history = {'fitness': [], 'best': []}
    
    st.subheader("Algorithm Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        population_size = st.slider("Population Size", 10, 50, 20)
        generations = st.slider("Number of Generations", 5, 50, 10)
    
    with col2:
        mutation_rate = st.slider("Mutation Rate", 0.0, 1.0, 0.1)
        crossover_rate = st.slider("Crossover Rate", 0.0, 1.0, 0.8)
    
    st.write("**Fitness Function:** Maximize sum of squares (Σx²)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Initialize Population", type="primary"):
            ga = GeneticAlgorithm(population_size, mutation_rate, crossover_rate, generations)
            st.session_state.ga = ga
            st.session_state.ga_population = ga.create_population(population_size)
            st.session_state.ga_phase = 1
            st.session_state.ga_generation = 0
            st.session_state.ga_history = {'fitness': [], 'best': []}
            st.rerun()
    
    with col2:
        if st.button("Next Phase") and st.session_state.ga_population is not None:
            st.session_state.ga_phase = (st.session_state.ga_phase % 5) + 1
            st.rerun()
    
    with col3:
        if st.button("Auto Run") and st.session_state.ga_population is not None:
            st.session_state.ga_auto_run = True
            st.rerun()
    
    if st.session_state.ga_population is not None:
        ga = st.session_state.ga
        population = st.session_state.ga_population
        phase = st.session_state.ga_phase
        generation = st.session_state.ga_generation
        
        phase_names = {
            1: "1. Initial Population",
            2: "2. Fitness Function",
            3: "3. Selection",
            4: "4. Crossover",
            5: "5. Mutation"
        }
        
        st.subheader(phase_names[phase])
        
        if phase == 1:  # Initial Population
            st.write("**Random chromosomes (initial population):**")
            
            # Show sample chromosomes
            sample_size = min(5, len(population))
            for i in range(sample_size):
                chrom = population[i]
                st.write(f"Chromosome {i+1}: {[f'{x:.2f}' for x in chrom[:5]]}... (fitness: {ga.fitness_function(chrom):.2f})")
            
            # Visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            fitness_scores = [ga.fitness_function(ind) for ind in population]
            ax.hist(fitness_scores, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Fitness Score', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Initial Population Fitness Distribution', fontsize=14, weight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        elif phase == 2:  # Fitness Function
            fitness_scores = [ga.fitness_function(ind) for ind in population]
            st.session_state.ga_fitness = fitness_scores
            
            st.write("**Fitness scores for all chromosomes:**")
            fitness_df = pd.DataFrame({
                'Chromosome': range(1, len(population) + 1),
                'Fitness': [f'{f:.2f}' for f in fitness_scores]
            })
            st.dataframe(fitness_df.head(10), use_container_width=True, hide_index=True)
            
            # Visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            ax1.bar(range(1, len(fitness_scores) + 1), fitness_scores, color='green', alpha=0.7)
            ax1.set_xlabel('Chromosome', fontsize=12)
            ax1.set_ylabel('Fitness Score', fontsize=12)
            ax1.set_title('Fitness Scores', fontsize=12, weight='bold')
            ax1.grid(True, alpha=0.3, axis='y')
            
            ax2.hist(fitness_scores, bins=15, color='coral', alpha=0.7, edgecolor='black')
            ax2.axvline(max(fitness_scores), color='red', linestyle='--', linewidth=2, label='Best')
            ax2.axvline(np.mean(fitness_scores), color='blue', linestyle='--', linewidth=2, label='Mean')
            ax2.set_xlabel('Fitness Score', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title('Fitness Distribution', fontsize=12, weight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        elif phase == 3:  # Selection
            if st.session_state.ga_fitness is None:
                st.session_state.ga_fitness = [ga.fitness_function(ind) for ind in population]
            
            fitness_scores = st.session_state.ga_fitness
            selected = []
            
            # Tournament selection
            for _ in range(population_size):
                tournament = random.sample(list(zip(population, fitness_scores)), 3)
                winner = max(tournament, key=lambda x: x[1])
                selected.append(winner[0])
            
            st.session_state.ga_selected = selected
            
            st.write("**Selected parents (highlighted in green):**")
            
            # Visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            colors = ['green' if ind in selected else 'lightblue' for ind in population]
            ax.bar(range(1, len(fitness_scores) + 1), fitness_scores, color=colors, alpha=0.7)
            ax.set_xlabel('Chromosome', fontsize=12)
            ax.set_ylabel('Fitness Score', fontsize=12)
            ax.set_title('Selection Phase (Green = Selected Parents)', fontsize=14, weight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            st.pyplot(fig)
        
        elif phase == 4:  # Crossover
            if not st.session_state.ga_selected:
                st.warning("Please run Selection phase first!")
                return
            
            selected = st.session_state.ga_selected
            offspring = []
            crossover_points = []
            
            for i in range(0, len(selected) - 1, 2):
                parent1 = selected[i]
                parent2 = selected[i + 1]
                child1, child2 = ga.crossover(parent1, parent2)
                offspring.extend([child1, child2])
                
                # Find crossover point
                point = random.randint(1, len(parent1) - 1) if random.random() <= ga.crossover_rate else -1
                crossover_points.append(point)
            
            st.session_state.ga_offspring = offspring
            
            st.write("**Offspring created through crossover:**")
            st.write(f"Sample crossover (showing first 2 pairs):")
            
            for i in range(min(2, len(crossover_points))):
                idx = i * 2
                p1 = selected[idx]
                p2 = selected[idx + 1]
                c1 = offspring[idx]
                c2 = offspring[idx + 1]
                point = crossover_points[i]
                
                st.write(f"**Pair {i+1}:**")
                st.write(f"  Parent 1: {[f'{x:.2f}' for x in p1[:5]]}...")
                st.write(f"  Parent 2: {[f'{x:.2f}' for x in p2[:5]]}...")
                if point > 0:
                    st.write(f"  Crossover point: {point}")
                st.write(f"  Child 1: {[f'{x:.2f}' for x in c1[:5]]}...")
                st.write(f"  Child 2: {[f'{x:.2f}' for x in c2[:5]]}...")
            
            # Visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            parent_fitness = [ga.fitness_function(ind) for ind in selected[:10]]
            offspring_fitness = [ga.fitness_function(ind) for ind in offspring[:10]]
            
            x = np.arange(min(len(parent_fitness), len(offspring_fitness)))
            width = 0.35
            ax.bar(x - width/2, parent_fitness[:len(x)], width, label='Parents', color='blue', alpha=0.7)
            ax.bar(x + width/2, offspring_fitness[:len(x)], width, label='Offspring', color='green', alpha=0.7)
            ax.set_xlabel('Pair Index', fontsize=12)
            ax.set_ylabel('Fitness Score', fontsize=12)
            ax.set_title('Crossover: Parents vs Offspring', fontsize=14, weight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            st.pyplot(fig)
        
        elif phase == 5:  # Mutation
            if not st.session_state.ga_offspring:
                st.warning("Please run Crossover phase first!")
                return
            
            offspring = st.session_state.ga_offspring
            mutated = []
            mutation_info = []
            
            for i, ind in enumerate(offspring):
                mut_ind, mut_indices = ga.mutate(ind)
                mutated.append(mut_ind)
                if mut_indices:
                    mutation_info.append((i, mut_indices))
            
            st.session_state.ga_mutated = mutated
            
            st.write("**Mutated chromosomes (mutations highlighted):**")
            if mutation_info:
                st.write(f"Found mutations in {len(mutation_info)} chromosomes")
                for chrom_idx, mut_indices in mutation_info[:5]:
                    st.write(f"  Chromosome {chrom_idx+1}: Mutated positions {mut_indices}")
            else:
                st.write("No mutations occurred in this generation")
            
            # Visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            offspring_fitness = [ga.fitness_function(ind) for ind in offspring]
            mutated_fitness = [ga.fitness_function(ind) for ind in mutated]
            
            ax1.scatter(range(len(offspring_fitness)), offspring_fitness, 
                       color='blue', alpha=0.6, label='Before Mutation', s=50)
            ax1.scatter(range(len(mutated_fitness)), mutated_fitness, 
                       color='red', alpha=0.6, label='After Mutation', s=50, marker='x')
            ax1.set_xlabel('Chromosome Index', fontsize=12)
            ax1.set_ylabel('Fitness Score', fontsize=12)
            ax1.set_title('Mutation Effect on Fitness', fontsize=12, weight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Update population and show evolution
            st.session_state.ga_population = mutated[:population_size]
            st.session_state.ga_generation += 1
            
            # Update history
            current_fitness = [ga.fitness_function(ind) for ind in mutated]
            st.session_state.ga_history['fitness'].append(np.mean(current_fitness))
            st.session_state.ga_history['best'].append(max(current_fitness))
            
            # Evolution graph
            if len(st.session_state.ga_history['best']) > 1:
                ax2.plot(st.session_state.ga_history['fitness'], 'b-', 
                        label='Average Fitness', linewidth=2)
                ax2.plot(st.session_state.ga_history['best'], 'r-', 
                        label='Best Fitness', linewidth=2)
                ax2.set_xlabel('Generation', fontsize=12)
                ax2.set_ylabel('Fitness Score', fontsize=12)
                ax2.set_title('Fitness Evolution Over Generations', fontsize=12, weight='bold')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.success(f"Generation {st.session_state.ga_generation} completed!")
            st.write(f"Best fitness so far: {max(st.session_state.ga_history['best']):.2f}")
            
            # Reset phase for next generation
            if st.session_state.ga_generation < generations:
                st.session_state.ga_phase = 1
                st.session_state.ga_fitness = None
                st.session_state.ga_selected = []
                st.session_state.ga_offspring = []
        
        # Show best solution
        if st.session_state.ga_history['best']:
            st.subheader("Best Solution")
            best_fitness = max(st.session_state.ga_history['best'])
            best_idx = st.session_state.ga_history['best'].index(best_fitness)
            best_gen = best_idx + 1
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Best Fitness", f"{best_fitness:.2f}")
            with col2:
                st.metric("Generation", best_gen)
            with col3:
                st.metric("Current Generation", st.session_state.ga_generation)
