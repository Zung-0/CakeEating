# CakeEating
Python code for solving the Cake Eating Problem using value function iteration. Inspired by Quantitative Economics (QuantEcon) by John Stachurski and Thomas Sargent. Roughly following chapters 2 and 3 of Economic Dynamics by Jerome Adda and Russell Cooper.

Relevant files (in \quantecon):

stochcp.py : Solves the Cake Problem. Works for the following cases: deterministic, stochastic endowment and functional discount factor. (The stochastic endowment can take a large -but finite- number of values).
discretecp.py: Solves the Discrete Choice (Eat or Wait) Cake Problem. Didn't include it in the 'stochcp.py' because it would have become too messy.
discretecp_additive.py : Same as the 'discretecp.py', except it uses a different way of including the stochastic factor. Written just for replicating figure 3.7 of 'Economic Dynamics'.

All comments are welcome
