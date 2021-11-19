import random

from nqueen import NQueen

def chromosome_generator(size):
	return random.sample(range(size), size)


def initial_population(n, population_size):
	return [NQueen(n, chromosome_generator(n))
		 for _ in range(population_size)]


def tournament_selection(population, tournament_size, parents_size):
	parents = random.choices(population, k=tournament_size)
	parents = sorted(parents, key=lambda agent: agent.fitness, reverse=True)
	return parents[:parents_size]


def ordered_breed(parent1, parent2):
	child = []
	childP1 = []
	childP2 = []

	geneA = int(random.random() * len(parent1.permutation))
	geneB = int(random.random() * len(parent1.permutation))

	start = min(geneA, geneB)
	end = max(geneA, geneB)

	childP1 = parent1.permutation[start:end]

	childP2 = [item for item in parent2.permutation if item not in childP1]

	child = NQueen(parent1.n, childP1 + childP2)

	return child


def atypical_breed(parent1, parent2):
	child = []
	
	not_selected = set(range(parent1.n))
	
	for i in range(parent1.n):
		if parent1.permutation[i] == parent2.permutation[i]:
			child.append(parent1.permutation[i])
			not_selected.remove(parent1.permutation[i])
		else:
			child.append(None)
	
	child1 = child[:]
	not_selected_tmp = set(not_selected)
	
	for i in range(parent1.n):
		if child1[i] is None:
			item = random.choice(list(not_selected_tmp))
			child1[i] = item
			not_selected_tmp.remove(item)
	
	child1 = NQueen(parent1.n, child1)

	child2 = child[:]
	not_selected_tmp = set(not_selected)

	for i in range(parent1.n):
		if child2[i] is None:
			item = random.choice(list(not_selected_tmp))
			child2[i] = item
			not_selected_tmp.remove(item)

	child2 = NQueen(parent1.n, child2)

	return child1, child2


def another_breed(parent1, parent2):
	permutation1 = parent1.permutation[:]
	permutation2 = parent2.permutation[:]
	for i in range(0, parent1.n):
		# if abs(permutation1[i-1] - permutation1[i]) == 1:
		# 	permutation1[i], permutation2[i] = permutation2[i], permutation1[i]
		# if abs(parent2.permutation[i-1] - parent2.permutation[i]) == 1:
		# 	permutation1[i], permutation2[i] = permutation2[i], permutation1[i]
		
		if i in parent1.indanger_queens:
			permutation1[i], permutation2[i] = permutation2[i], permutation1[i]
		if i in parent2.indanger_queens:
			permutation1[i], permutation2[i] = permutation2[i], permutation1[i]
	
	return NQueen(parent1.n, permutation1), NQueen(parent1.n, permutation2)
	


def breed_population(parents, population_size):
	children = []

	# rate = int((population_size/len(parents))*2)
	rate = int((population_size/len(parents)))
	
	for _ in range(rate):
		random.shuffle(parents)
		for i in range(int(len(parents)/2)):
			two_child = another_breed(parents[i], parents[len(parents)-i-1])
			# two_child = atypical_breed(parents[i], parents[len(parents)-i-1])
			children.append(two_child[0])
			children.append(two_child[1])

	return children

def mutate(individual, mutation_rate):
	permutation = individual.permutation[:]
	if(random.random() < mutation_rate):
		while True:
			indanger = set(individual.indanger_queens)
			swapped = random.choice(list(indanger))
			indanger.remove(swapped)
			swapWith = random.choice(list(indanger))
			if swapped == swapWith:
				continue
			
			gene1 = permutation[swapped]
			gene2 = permutation[swapWith]
	
			permutation[swapped] = gene2
			permutation[swapWith] = gene1
			return NQueen(individual.n, permutation)
	else:
		return None


# def mutate(individual, mutation_rate):
# 	if random.random() < mutation_rate:
# 		return NQueen(individual.n, chromosome_generator(individual.n))

# 	return individual


def mutate_population(population, mutation_rate):
	mutated_children = []

	for ind in range(0, len(population)):
		mutatedInd = mutate(population[ind], mutation_rate)
		if mutatedInd is not None:
			mutated_children.append(mutatedInd)

	return mutated_children


def replacement(children, current_population):
	children = children + current_population

	unique_children = []
	unique_children_hash = []
	
	
	for child in children:
		if hash(child) in unique_children_hash:
			continue
		else:
			unique_children.append(child)
			unique_children_hash.append(hash(child))

	children = unique_children

	children = sorted(
		list(children), key=lambda agent: agent.fitness, reverse=True)
	
	children = children[:len(current_population)]

	return children


def generate_neighbor(individual):
	permutation = individual.permutation
	while True:
		swapped = int(random.random() * len(permutation))
		swapWith = int(random.random() * len(permutation))
		
		if swapped == swapWith:
			continue
		
		gene1 = permutation[swapped]
		gene2 = permutation[swapWith]

		permutation[swapped] = gene2
		permutation[swapWith] = gene1
		yield NQueen(individual.n, permutation)
		permutation = individual.permutation

def local_search(individual, max_depth, max_count):
	best = individual
	
	generator = generate_neighbor(individual)
	
	for _ in range(max_depth):
		neighbor = next(generator)
		count = 0
		while best.fitness > neighbor.fitness and count < max_count:
			neighbor = next(generator)
			count += 1
		
		if count != max_count:
			best = neighbor
	
	# print(individual.fitness, best.fitness)
	
	return best
	

def improve(population, improve_rate, max_depth, max_count):
	
	improved_population = []
	
	for individual in population:
		if random.random() < improve_rate:
			improved_population.append(local_search(individual, max_depth, max_count))
	
	return improved_population

def evaluate(population):
	pop_fitness = sorted(
		[agent.fitness for agent in population], reverse=True)

	return sum(pop_fitness), pop_fitness[0], pop_fitness[5], pop_fitness[30], pop_fitness[-1]


def generate_genration(epoch, previous_population, tournament_size, parents_size, improve_rate, max_depth, max_count, mutation_rate):
	parents = tournament_selection(
		previous_population, tournament_size, parents_size)

	children = breed_population(parents, len(previous_population))
	
	children = improve(children, improve_rate, max_depth, max_count)
	
	mutated_children = mutate_population(children, mutation_rate)
	
	children = children + mutated_children
	
	mutated_children = mutate_population(children, mutation_rate)

	children = children + mutated_children

	next_population = replacement(children, previous_population)

	eval_ = evaluate(next_population)

	print("Epoch", epoch, ":\tPopulation total fitness:",
			eval_[0], "\tBest fitness:", eval_[1], eval_[2], eval_[3], eval_[4])

	return next_population


def NQueens_MA(n, n_generations, population_size, tournament_size, parents_size, improve_rate, max_depth, max_count, mutation_rate):
	population = initial_population(n, population_size)
	
	for i in range(n_generations):
		population = generate_genration(
			i, population, tournament_size, parents_size, improve_rate, max_depth, max_count, mutation_rate)
		
		if population[0].fitness == 0:
			break
		

	print('Best Answer:')
	print(population[0].permutation)
