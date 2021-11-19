from utils import NQueens_MA

n = 100
n_generations = 1000
population_size = 500
tournament_size = 400
parents_size = 200
improve_rate = 0.7
max_depth = 2
max_count = 10
mutation_rate = 1

NQueens_MA(n, n_generations, population_size, tournament_size,
           parents_size, improve_rate, max_depth, max_count, mutation_rate)
