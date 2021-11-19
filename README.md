# TSP Problem with Genetic Algorithm
## Algorithm
Here we implemented our chromosomes as permutations, so the chromosome space already satisfy two constraints (row and column attacks).

For fitness function we consider the number of pair attacks occurring in the permutations. Also, note that fitness function is the negative of this number. Therefore, reaching zero is the goal in this problem.

First, we generate our initial population with random permutations.

The main process is a loop with several sections. At the end of every iteration in this loop, a new generation is born, preferably with better individuals.

Here is each section of this process:

### Selection
The first thing we do with the previous generation is to select a subset of them as our parents. The primary consideration here is to make a balance between elites and vulgar to prevent overfitting or underfitting.

The algorithm used for this matter is tournament selection. For this, we choose a subset of the population by random, then we sort them based on their fitness and select a number of them as parents.

### Breeding
With the parents being chosen, it is time for mating them with each other. Two individuals create a child with ordered recombination.

For choosing each pair, we shuffle parents and mate them by random. We repeat this shuffling until the number of our children has reached the population size.

### Local-search (improving)
This is the part that makes this algorithm an MA. Here we review children based on a rate called `improve_rate`. These children are considered for being checked for being replaced by one of their neighbors. We consider a permutation as neighbor if they only differ in a few items in the permutation. We find a neighbor by randomly choosing two genes to be swapped with each other. Note that we only replace a permutation, if it has a better fitness. Also, we only look for better neighbor a limited time specified by `max_count`. We do this local search a number of times which is identified by `max_depth`.

These three parameters, `improve_rate`, `max_count`, and `max_depth`, impact the speed and performance of the algorithm dramatically.

### Mutation
We have a parameter called the `mutation_rate`, which specifies the probability of mutation. But in this problem because of the issues of local optimum we have set this variable to one. Also, we run the mutation twice.

The mutations in this problem are done by swapping two cities(genes) in a permutation.

### Replacement
The final step in the loop is to specify the next generation. We mix both mutated children, improved original children, and original population. Then we sort them by fitness and choose the best ones to create our next generation.


## Test
### n = 100
In this case after many trials. I understood that I need to increase exploration due to stucking in local optimums. Also, a large number of generations is required to give a chance to algorithm to find a good answer.

## Considerations
- Because we want to optimize our answers based on the total distance of the cycle, our fitness function is `1/total_distance` (so we try to maximize fitness).
