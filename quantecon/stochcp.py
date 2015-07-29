"""
Filename: stochcp.py

Authors: Diego Zuniga and Alex Carrasco

Solving the stochastic cake problem via value function iteration.

Based on 'Quantitative Economics with Python'* by John Starchuski and Thomas Sargent

*Following the template of the growth model class in optgrowth.py
*Located at http://quant-econ.net
"""

from __future__ import division  # Omit for Python 3.x
from textwrap import dedent
import numpy as np
from scipy.optimize import fminbound
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import norm
from compute_fp import compute_fixed_point
import matplotlib.pyplot as plt

class StochCakeProblem(object):
	"""

	This class defines the primitives representing the stochastic cake problem.

	Parameters
	----------
	beta : scalar(int), optional(default=.95)
		The utility discounting parameter
	return : scalar(int), optional(default=1/.95)
		The return on leftover cake.
	u : function, optional(default = (c**(1 - gamma))/(1 - gamma)))
		The utility function.  Default is CRRA with parameter gamma
	gamma : scalar(float), optional(default=2)
	logutil : Boolean, optional(default=False)
		Transforms the utility function into log utility
	deterministic : Boolean, optional(default=True)
		Eliminates the uncertainty
	shocks : list or array, optional(default= [.9, 1.1])
		Size of the shocks.
	transition : list or array, optional(default=[[.5, .5], [.5, .5]])
		Transition matrix between possible shocks
	functional : Boolean, optional(default = False)
		Allows for a functional discount factor
	grid_min : scalar(int), optional(default=.4)
		The minimum grid value
	grid_max : scalar(int), optional(default=2)
		The maximum grid value
	grid_size : scalar(int), optional(default=150)
		The size of grid to use.

	Attributes
	----------
	beta, return, u  : see Parameters
	grid : array_like(float, ndim=1)
		The grid over possible cake size.

	"""
	def __init__(self, beta=0.95, r = 1/0.95, gamma = 2, logutil=False, functional=False,
				 shocks = [.9, 1.1], transition = [[.5,.5],[.5,.5]], deterministic=False, grid_max=2,
				 grid_min = .4, grid_size=150):

		self.r, self.beta, self.gamma = r, beta, gamma
		self.shocks = shocks
		self.transition = transition
		self.functional, self.deterministic = functional, deterministic
		
		if functional == True:
			self.beta = lambda c: norm.cdf(1.65 - c) 
		if deterministic == True:
			self.transition = np.asarray([[1]])
			self.shocks = np.asarray([0])

		if logutil == False:
			self.u = lambda c: (c**(1 - gamma))/(1 - gamma)
		else:
			self.u = np.log
			self.gamma = 1
		self.grid = np.linspace(grid_min, grid_max, grid_size)

		if len(self.transition) != len(self.shocks):
			self.transition = [[1/len(self.shocks)]*len(self.shocks)]*len(self.shocks)

		for j in xrange(len(self.transition)):
			self.transition[j] = [max(k, 0) for k in self.transition[j]]
			self.transition[j] /= np.asarray(list(self.transition[j])).sum()
			if np.asarray(self.transition[j]).sum() == 0:
				self.transition[j] = np.asarray([1/len(self.transition)]*len(self.transition))
		self.transition = np.asarray(self.transition)

	def __repr__(self):
		m = "CakeProblem(beta={b}, return={r}, grid_min={gmin} grid_max={gm}, grid_size={gs})"
		return m.format(b=self.beta, r=self.r, gmin=self.grid.min(), gm=self.grid.max(), gs=self.grid.size)

	def __str__(self):
		m = """\
		CakeEatingModel:
		  - beta (discount factor)                             : {b}
		  - r (return factor)                                  : {r}
		  - u (utility function)                               : {u}
		  - y (stochastic endowments - range)                  : ({yl}, {yh})
		  - transition (transition matrix for endowments)      : {trans}
		  - grid bounds (bounds for grid over savings values)  : ({gl}, {gm})
		  - grid points (number of points in grid for savings) : {gs}
		"""
		return dedent(m.format(b=self.beta, r=self.r, u=self.u,
							   yl=self.shocks[0], yh=self.shocks[-1], trans=self.transition,
							   gl=self.grid.min(), gm=self.grid.max(),
							   gs=self.grid.size))

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
		
		Aw = [InterpolatedUnivariateSpline(self.grid, w[i], k=3) for i in xrange(len(self.shocks))]
		Awx = lambda y : [Aw[i](y[i]) for i in xrange(len(y))]
		#Awx = lambda y: np.asarray([function(y) for function in Aw]) 

		if compute_policy:
			sigma = np.asarray([np.empty(len(w[0]))]*len(self.shocks))

		Tw = np.asarray([np.empty(len(w[0]))]*len(self.shocks))
		
		if self.functional: 
			# It is necessary to formulate the objective function differently for the functional form
			for i, x in enumerate(self.grid):
				for j in xrange(len(self.shocks)):
					objective = lambda c: - self.u(c) - self.beta(c) * (np.dot(self.transition[j], Awx(np.add(self.r*(x - c), self.shocks))))
					c_star = fminbound(objective, 1e-6, x-1e-6)
			
					if compute_policy:
						sigma[j][i] = c_star
					Tw[j][i] = - objective(c_star)

		else: 
			for i, x in enumerate(self.grid):
				for j in xrange(len(self.shocks)):
					objective = lambda c: - self.u(c) - self.beta * (np.dot(self.transition[j], Awx(np.add(self.r*(x - c), self.shocks))))
					#def objective(c):
					#	u = self.u(c)
					#	expectation = np.dot(self.transition[j], Awx(self.r*(x - c) + self.shocks)))
					#	return (- u - self.beta*expectation)

					c_star = fminbound(objective, 1e-6, x-1e-6)

					if compute_policy:
						sigma[j][i] = c_star
					Tw[j][i] = - objective(c_star)

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

	#### Policy function iteration - Work in Progress - Z

	def value(self, sigma, b_min=.01):
		"""
		Compute the value function obtained by 
		operating policy sigma forever

		Parameters
		----------
		sigma : array_like(float, ndim=1)
				The policy function for different grid points
				(Amount of cake to be eaten as a function of cake size)

		Returns:
		--------
		w : array_like(float, ndim=1)
			The value function associated with sigma 
		"""

		sigma_function = InterpolatedUnivariateSpline(self.grid, sigma, k=1)
		w = np.empty(self.grid.size)
		for i, x in enumerate(self.grid):
			t = 0
			b = 1
			new_x = x
			c = sigma_function(new_x)
			while b > b_min:
				w[t] += b*self.u(c)
				t = t + 1
				b = self.beta*b
				new_x = self.r*(x - c) #Transition law
				c = sigma_function(self.r*(x - c))
				if c < 1e-6: #break #Maybe 
					b = b_min 

		return w 

	def policy(self, w):
		"""
		Compute the optimal policy for a 
		given value function

		Parameters:
		-----------
		w : array_like(float, ndim=1)
			A value function for every grid point

		Returns:
		--------
		sigma : array_like(float, ndim=1)
			The optimal policy for w
		"""
		Awx = InterpolatedUnivariateSpline(self.grid, w, k=1) 
		sigma = np.empty(self.grid.size)
		for i, x in enumerate(self.grid):
			#Iterative process for finding the optimal consumption, slower than using fminbound
			c_star = x/2
			cgrid = np.linspace(1e-6, x - 1e-6, self.grid.size)
			objective = lambda c: self.u(c) + self.beta * self.prob * Awx(self.r*(x - c) + self.yl) + self.beta * (1 - self.prob) * Awx(self.r * (x - c) + self.yh)
			for c in cgrid:
				if objective(c) > objective(c_star):
					c_star = c
			
			sigma[i] = c_star
			
		return sigma