import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class PSO:
	"""
	Particle Swarm optimization class
	"""
	def __init__(self,opt_func, opt_task,  num_particles, num_dims, constraints, device ,plot = False, animate = False):
		"""
		Constructor for PSO class

		Inputs:
		param opt_func : Passes the function object which has to optimized
		param opt_task : "min" or "max"
		param num_particles: specifies number of particles in PSO
		param num_dims: Specifies the dimensions of space in which the PSO particles exist. Must be equal to the number of independent variables in the opt_func
		param constraints: Specifies constraints on search space. Specified as list of lists, where the latter are 2-element [min,max] lists.
				Example  : [[-10,10],[-15,15]] where -10 is min for first indp. var., 10 is max for first indp. var. and so on.
		param device    : Specifies wheter to run on cpu or gpu ("cuda") 
		param plot      : Boolean to specify wheter to plot gbest fitness vs iterations plot.
		param animate   : Boolean to specify wheter to create a video of best particle.  
		"""
		#Checking if gpu is available. If not, cpu will be used.
		if device == "gpu":
			device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.device = device
		print(f"Using {device}")

		self.opt_func = opt_func
		self.opt_task = opt_task
		self.num_particles = num_particles
		self.num_dims = num_dims
		self.constraints  = torch.Tensor(constraints).T.to(device)
		self.particle_pos = self.constraints[0,:] + torch.rand(num_particles, num_dims).to(device)*(self.constraints[1,:] - self.constraints[0,:])
		self.particle_vels = torch.randn(num_particles, num_dims).to(device)
		self.particle_pbests = self.particle_pos
		self.gbest = torch.randn(num_dims).to(device)
		self.particle_fitness = torch.randn(num_particles).to(device)
		self.particle_best_fitness = self.particle_fitness
		self.best_gbest_fitness = 2
		self.plot = plot
		self.animate = animate
		if self.plot:
			self.gbest_fitness_lst = []
		if self.animate:
			self.best_particles_lst = []


	def _update_anim(self, index, scat,ax):
		"""
		Update function for creating animation.

		Input: 
		param  index: frame number
		param  scat : scatter plot object to update particle location
		param  ax   : axes object to update axes title
		"""
		scat.set_offsets(([self.best_particles_lst[index][0], self.best_particles_lst[index][1]]))
		ax.set_title(f"Iteration {index}")


	def _animate(self, num_iterations):
		"""
		Creates and saves the animation of best particle as a gif.

		Inputs:
		param num_iterations: number of iterations which PSO took to converge
		"""
		x = np.linspace(self.constraints[0, 0], self.constraints[1,0], 100)
		y = np.linspace(self.constraints[0, 1], self.constraints[1,1], 100)
		X, Y = np.meshgrid(x, y)
		Z = self.opt_func([torch.Tensor(X), torch.Tensor(Y)])
		fig = plt.figure()
		ax = plt.axes(xlim=(self.constraints[0, 0], self.constraints[1,0]), ylim=(self.constraints[0, 1], self.constraints[1,1]))
		ax.contour(X,Y,Z)
		scat = ax.scatter(self.best_particles_lst[0][0],self.best_particles_lst[0][1],  c="r")
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_title("Iteration 0")
		anim = FuncAnimation(fig, self._update_anim, fargs = (scat,ax), interval = 100)
		anim.save("new_video.gif")


	def _get_fitness(self):
		"""
		Computes fitness values for all PSO particles
		"""
		self.particle_fitness = self.opt_func([self.particle_pos[:,dim] for dim in range(self.num_dims)])

	def _update_bests(self, iteration):
		"""
		Updates the pbests for all particles and the gbest.

		Inputs:
		param iteration: The current iteration number
		"""
		if iteration == 1:
			self.particle_best_fitness = self.particle_fitness
			self.particle_pbests = self.particle_pos
			if self.opt_task == "max":
				fitness_val, index = self.particle_best_fitness.max(0)
			else:
				fitness_val, index = self.particle_best_fitness.min(0)
			if self.animate:
				self.best_particles_lst.append(self.particle_pos[index])
			self.best_gbest_fitness = fitness_val
			self.gbest = self.particle_pbests[index]
		else:
			if self.opt_task == "max":
				mask1 = (self.particle_best_fitness > self.particle_fitness).type(torch.int)
				mask2 = (self.particle_best_fitness < self.particle_fitness).type(torch.int)
				self.particle_best_fitness = mask1*self.particle_best_fitness + mask2*self.particle_fitness
				self.particle_pbests = mask1.reshape((self.num_particles, 1))*self.particle_pbests + mask2.reshape((self.num_particles, 1))*self.particle_pos
				fitness_val, index = self.particle_best_fitness.max(0)
				if self.animate:
					self.best_particles_lst.append(self.particle_pos[index])
				if fitness_val > self.best_gbest_fitness:
					self.best_gbest_fitness = fitness_val
					self.gbest = self.particle_pbests[index]
			else:
				mask1 = (self.particle_best_fitness < self.particle_fitness).type(torch.int)
				mask2 = (self.particle_best_fitness > self.particle_fitness).type(torch.int)
				self.particle_best_fitness = mask1*self.particle_best_fitness + mask2*self.particle_fitness
				self.particle_pbests = mask1.reshape((self.num_particles, 1))*self.particle_pbests + mask2.reshape((self.num_particles, 1))*self.particle_pos
				fitness_val, index = self.particle_best_fitness.min(0)
				if self.animate:
					self.best_particles_lst.append(self.particle_pos[index])
				if fitness_val < self.best_gbest_fitness:
					self.best_gbest_fitness = fitness_val
					self.gbest = self.particle_pbests[index]

	def solve(self, w, c1, c2, num_iter = 1000):
		"""
		Runs the PSO on opt_sunc within constraints to perform opt_task

		Inputs:
		param  w : weight of velocity of particle
		param c1 : weight of velocity of particle towards its pbest
		param c2 : weight of velocity of particle towards gbest

		Outputs:
		2-tuple : (optimized solution or position of best particle, fitness of best particle)
		"""
		iteration = 1
		while( (self.particle_pos[0] == self.particle_pos).all().item() == False or iteration <= num_iterations):
			self._get_fitness()
			self._update_bests(iteration)
			#Updating velocities of particles
			self.particle_vels = w*self.particle_vels + c1*torch.rand(1).to(self.device)*(self.particle_pbests - self.particle_pos) + c2*torch.rand(1).to(self.device)*(self.gbest - self.particle_pos)
			
			#Updating positions of particles
			self.particle_pos = self.particle_pos + self.particle_vels

			# Clamping positions of particles if they exit the constrained search space.
			for dim in range(self.num_dims):
				torch.clamp_(self.particle_pos[:, dim], min = self.constraints[0, dim], max = self.constraints[1, dim])

			# Gathering values for plotting
			if self.plot:
				self.gbest_fitness_lst.append(self.best_gbest_fitness)

			# Printing per iteration log
			print(f"Iteration {iteration} : Gbest = {self.gbest}, Gbest_fitness : {self.best_gbest_fitness}")
			print("__________________________________________________________________________________")
			iteration += 1

		# PLotting gbest fitness vs iteration number 
		if self.plot:
			plt.plot(range(iteration-1), self.gbest_fitness_lst)
			plt.xlabel("Iteration Number")
			plt.ylabel("Fitness Value")
			plt.xscale("log")
			plt.savefig("new_plot.pdf")

		# Animating the best particle
		if self.animate:
			self._animate(iteration-1)

		
		return self.particle_pos[0], self.opt_func([self.particle_pos[0, dim] for dim in range(self.num_dims)])




			
		




