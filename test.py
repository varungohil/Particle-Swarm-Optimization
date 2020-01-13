from pso.pso import PSO

def square(x):
	return x[0]**2

p = PSO(square, "min", 10, 1, [[-4, 4]])
p.solve(1, 0.8, 0.8)