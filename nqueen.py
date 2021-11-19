class NQueen:
	
	def __init__(self, n, permutation):
		self.n = n
		self.permutation = permutation
		
		self.indanger_queens = set()
		
		self.fitness = None
		self.get_fitness()
		
	def get_fitness(self):
		if self.fitness is not None:
			return self.fitness
		
		self.fitness = 0
		
		for i in range(self.n - 1):
			for j in range(i + 1, self.n):
				if abs(i - j) == abs(self.permutation[i] - self.permutation[j]):
					self.fitness -= 1
					self.indanger_queens.add(i)
					self.indanger_queens.add(j)
		
		return self.fitness
