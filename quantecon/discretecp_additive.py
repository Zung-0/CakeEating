"""
Filename: discretecp.py

Authors: Diego Zuniga and Alex Carrasco

Solving the stochastic cake problem via value function iteration.
Discrete Choice Case.

Based on:
'Quantitative Economics with Python'* by John Starchuski and Thomas Sargent
'Dynamic Economics' by Jerome Adda and Russel W. Cooper

*Following the template of the growth model class in optgrowth.py
*Located at http://quant-econ.net
"""

from __future__ import division  # Omit for Python 3.x
from textwrap import dedent
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from compute_fp import compute_fixed_point
import matplotlib.pyplot as plt

class DiscreteCakeProblem_additive(object):
	"""

	This class defines the primitives representing the cake problem.

	Parameters
	----------
	beta : scalar(int), optional(default=.95)
		The utility discounting parameter
	return : scalar(int), optional(default=.95)
		The shrink rate on leftover cake.
	u : function, optional(default = (c**(1 - gamma))/(1 - gamma)))
		The utility function.  Default is CRRA with parameter gamma
	gamma : scalar(float), optional(default=2)
	logutil : Boolean, optional(default=False)
		Transforms the utility function into log utility
	shocks : list(float), optional(default= .8, 1.2)
		Size of the shocks.
	transition : list(float), optional(default=identity matrix)
		Transition matrix between possible shocks
	grid_min : scalar(int), optional(default=.4)
		The minimum grid value
	grid_max : scalar(int), optional(default=2)
		The maximum grid value
	grid_size : scalar(int), optional(default=150)
		The size of grid to use.

	Attributes
	----------
	beta, r, u  : see Parameters
	grid : array_like(float, ndim=1)
		The grid over possible cake size.

	"""
	def __init__(self, beta=0.95, r = 0.95, gamma = 2, logutil=False,
				 transition = [[1, 0], [0, 1]], shocks=[.8, 1.2],
				 grid_max=2, grid_min = .4, grid_size=150):

		self.r, self.beta, self.gamma = r, beta, gamma
		self.transition = transition
		self.shocks = shocks

		if logutil == False:
			self.u = lambda c: (c**(1 - gamma))/(1 - gamma)
		else:
			self.u = np.log
			self.gamma = 1
		self.grid = np.linspace(grid_min, grid_max, grid_size)
		
		# Validating the transition matrix (Make sure it covers all posible shocks)
		## Z: I'm torn between raising an error (as I've done) or quietly revert to the identity transition
		## matrix or a uniform distribution matrix (same prob to every outcome). 
		if len(self.transition) != len(self.shocks):
			#raise ValueError('Transition matrix does not correspond to the number of possible shocks.')
			#self.transition = np.identity(len(self.shocks))
			self.transition = [[1/len(self.shocks)]*len(self.shocks)]*len(self.shocks)

		# All rows must add up to one (as they are a probability distribution)
		## Z: I could raise an error here, too.
		## Or convert all negative probabilities to zero and normalize all probabilities to one.
		## In case the transition matrix is made of zeroes, use the identity matrix or a uniform distribution.
		for j in xrange(len(self.transition)):
			self.transition[j] = [max(k, 0) for k in self.transition[j]]
			self.transition[j] /= np.asarray(list(self.transition[j])).sum()
			if np.asarray(self.transition[j]).sum() == 0:
				#raise ValueError('Transition matrix probabilities wrongly specified.')
				self.transition[j] = np.asarray([1/len(self.transition)]*len(self.transition))
				#self.transition = np.identity(len(self.shocks))
		self.transition = np.asarray(self.transition)
	def __repr__(self):
		m = "CakeProblem(beta={b}, shrink={r}, grid_min={gmin} grid_max={gm}, grid_size={gs})"
		return m.format(b=self.beta, r=self.r, gmin=self.grid.min(), gm=self.grid.max(), gs=self.grid.size)

	def __str__(self):
		m = """\
		CakeEatingModel:
		  - beta (discount factor)                             : {b}
		  - r (shrink factor)                                  : {r}
		  - u (utility function)                               : {u}
		  - grid bounds (bounds for grid over savings values)  : ({gl}, {gm})
		  - grid points (number of points in grid for savings) : {gs}
		"""
		return dedent(m.format(b=self.beta, r=self.r, u=self.u,
							   gl=self.grid.min(), gm=self.grid.max(),
							   gs=self.grid.size))

	def prob(self, shock_i, shock_j): #NO LONGER NECESSARY
		"""
		Returns the probability of observing shock j conditional in
		observing shock i. Assumes iid.
		Uses the previously specified transition matrix.

		Parameters:
		shock_i, shock_j : (int) (0, 1 , ...)
							Value of the shocks
		"""
		return self.transition[shock_i][shock_j]


	def bellman_operator(self, w, compute_policy=False):
		"""
		The approximate Bellman operator, which computes and returns the
		updated value function Tw on the grid points.

		Parameters
		----------
		w : array_like(float, ndim=1)
			The value of the input function on different grid points
		compute_policy : Boolean, optional(default=False)
			Whether or not to compute policy function

		"""
		# === Apply interpolation to w === #
		Aw = [InterpolatedUnivariateSpline(self.grid, w[i], k=3) for i in xrange(len(self.shocks))]
		Awx = lambda y: np.asarray([function(y) for function in Aw]) # Useful for computing the utility of waiting.
																	# Z: I think this way might be faster, haven't tested it yet.
		
				
		if compute_policy:
			sigma = np.asarray([np.empty(len(w[0]))]*len(self.shocks))
		
		Tw = np.asarray([np.empty(len(w[0]))]*len(self.shocks))
		
		for i, x in enumerate(self.grid):
			for j, e in enumerate(self.shocks):
				u_now = self.u(e*x)
				## Two possible ways of calculating the utility of waiting:
				u_wait  = self.beta*np.dot(self.transition[j], Awx(self.r*x))
				#u_wait = 0
				#for k in xrange(len(self.shocks)):
				#	u_wait += self.beta*self.prob(j, k)*Aw[k](self.r*x)
				Tw[j][i] = max(u_now, u_wait)

				if compute_policy:
					# The decision rule is the size of the shock that makes
					# the agent indifferent between eating and waiting
					sigma[j][i] = np.exp(u_wait - self.u(x))
					

		if compute_policy:
			return Tw, sigma
		else:
			return Tw


	def compute_greedy(self, w):
		"""
		Compute the w-greedy policy on the grid points.

		Parameters
		----------
		w : array_like(float, ndim=1)
			The value of the input function on different grid points

		"""
		Tw, sigma = self.bellman_operator(w, compute_policy=True)
		return sigma

	def plot(self):
		print self
		#w_guess = np.asarray([np.empty(len(self.grid))]*len(self.shocks))
		#w_guess = np.asarray([np.zeros(len(self.grid))]*len(self.shocks))
		w_guess = np.asarray([np.log(self.grid)]*len(self.shocks))
		#w_guess = np.asarray([self.grid]*len(self.shocks)) #Takes Longer.
		w_star = compute_fixed_point(self.bellman_operator, w_guess, max_iter=100000, verbose=0, error_tol=1e-3, print_skip=100)
		sigma_star = self.compute_greedy(w_star)

		fig, ax = plt.subplots(2, 1, figsize =(8,10))
		ax[0].set_xlabel("Cake Size")
		ax[1].set_xlabel("Cake Size")
		ax[0].set_ylabel("Value Function")
		ax[1].set_ylabel("Threshold shock")

		eating = [self.u(self.grid)*self.shocks[i] - .1  for i in xrange(len(self.shocks))] # Value function if always eating.
		                                                                           # Not labeled because they are ordered

		ax[0].set_color_cycle(['r', 'g', 'b', 'c', 'm', 'y'])
		labels = [r'Shock {0:.2g}'.format(self.shocks[k]) for k in xrange(len(self.shocks))] # Label "Shock 1", "Shock 2"
		for i in xrange(len(self.shocks)):
		    ax[0].plot(self.grid, w_star[i], lw = 2, alpha = .8, label= labels[i])
		    
		ax[1].plot(self.grid, sigma_star[0], 'k-', lw = 2, alpha = .8)
		ax[0].legend(loc = 'lower right')
		t = 'Discrete Choice - Cake Problem'
		fig.suptitle(t, fontsize=18)

		plt.show()