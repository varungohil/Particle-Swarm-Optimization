import torch
import numpy as np

class PSO:
	def __init__(self,opt_func, opt_task,  num_particles, num_dims, constraints):
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.opt_func = opt_func
		self.opt_task = opt_task
		self.num_particles = num_particles
		self.num_dims = num_dims
		self.constraints  = torch.Tensor(constraints).T.to(device)
		self.particle_pos = self.constraints[0,:] + torch.rand(num_particles, num_dims).to(device)*(self.constraints[1,:] - self.constraints[0,:])
		self.particle_vels = torch.randn(num_particles, num_dims).to(device)
		self.particle_pbests = self.particles_pos
		self.gbest = torch.randn(num_dims).to(device)
		self.particle_fitness = torch.randn(num_particles).to(device)
		self.particle_best_fitness = self.particle_fitness
		self.best_gbest_fitness = 0

	def _get_fitness(self):
		self.particle_fitness = self.opt_func([self.particles_pos[:,dim] for dim in range(self.num_dims)])

	def _update_bests(self, iteration):
		if iteration == 1:
			self.particle_best_fitness = self.particle_fitness
		else:
			mask1 = (self.particle_best_fitness > self.particle_fitness).type(torch.int)
			mask2 = (self.particle_best_fitness < self.particle_fitness).type(torch.int)
			self.particle_best_fitness = mask1*self.particle_best_fitness + mask2*self.particle_fitness
			self.particle_pbests = mask1.reshape((self.num_particles, 1))*self.particle_pbests + mask2.reshape((self.num_particles, 1))*self.particle_pos
		if self.opt_task == "max":
			fitness_val, index = self.particle_best_fitness.max(0)
		else:
			fitness_val, index = self.particle_best_fitness.min(0)
		if iteration == 1 or fitness_val > self.best_gbest_fitness:
			self.best_gbest_fitness = fitness_val
			self.gbest = self.particle_pbests[index]


	def solve(self, w, c1, c2):
		iteration = 1
		while( (self.particle_pos[0] == self.particle_pos).all().item() == False):
			self._get_fitness()
			self._update_bests()
			self.particle_vels = w*self.particle_vels + c1*torch.rand(1)*(self.particle_pbests - self.particles_pos) + c2*torch.rand(1)*(self.gbest - self.particle_pos)
			self.particle_pos = self.particle_pos + self.particle_vels
			iteration += 1
		return self.particle_pos[0], self.opt_func([self.particle_pos[0, dim] for dim in range(self.num_dims)])



			
		




