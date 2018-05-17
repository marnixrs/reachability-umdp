"""
Author: Marnix Suilen, march 2018
"""
from gpkit import Variable, Model
from gpkit.constraints.tight import Tight

# Parameters
delta = 0.2
Tight.reltol = 0.001

# Optimization variables
sigma1a = Variable("sigma1a")
sigma1b = Variable("sigma1b")
sigma2a = Variable("sigma2a")
sigma2b = Variable("sigma2b")
sigma3a = Variable("sigma3a")
sigma3b = Variable("sigma3b")
p1 = Variable("p1")
p2 = Variable("p2")
p3 = Variable("p3")


# Constraints
constraints = [p1 <= 0.429 * delta + 0.25 * (1 - delta),
               Tight(
                   [sigma1a + sigma1b <= 1,
                    sigma2a + sigma2b <= 1,
                    sigma3a + sigma3b <= 1]),
               p3 == 1,
               (sigma1a * 0.3 + sigma1b * 0.2 + (sigma1a * 0.3 + sigma1b * 0.2) * p1) / p1 <= 1]

# Objective function
objective = 1 / sigma1a + 1 / sigma1b + 1 / sigma2a + 1 / sigma2b + 1 / sigma3a + 1 / sigma3b

# Formulate the Model
m = Model(objective, constraints)

# Solve the Model and print the results table
print (m.solve(verbosity=0).table())
