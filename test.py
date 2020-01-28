from pso.pso import PSO
import numpy as np
import torch


def cross_table(x):
  return -0.0001*(torch.abs(torch.sin(x[0])*torch.sin(x[1])*torch.exp(torch.abs(100 - (torch.sqrt(x[0]**2 + x[1]**2)/np.pi) ).double())) + 1)**0.1


p = PSO(cross_table, "min", 10, 2, [[-10, 10], [-10, 10]], "cpu", True, True)

print(p.solve(0.8, 0.5, 0.3))