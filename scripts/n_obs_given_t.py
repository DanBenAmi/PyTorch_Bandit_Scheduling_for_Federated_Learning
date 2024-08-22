import numpy as np
from scipy.optimize import least_squares

# Given values (example values, replace with your actual values)
mu_values = np.linspace(0.1, 0.9, 3)
m = 25
t = 1e7
n = 500

# Define the system of equations
def equations(vars):
    x = vars
    common_expr = mu_values[0] + np.sqrt(((m + 1) * np.log(t)) / x[0])

    eqs = []
    # Common equality for all mu + sqrt(((m+1)*ln t)/x_i)
    for i in range(1, len(x)):
        eqs.append(mu_values[i] + np.sqrt(((m + 1) * np.log(t)) / x[i]) - common_expr)

    # Linear sum constraint
    eq_sum = weights @ x - (m * t / n)
    eqs.append(eq_sum)

    return eqs

# Initial guesses for x1 to x3
initial_guesses = [20000, 20000, 20000000]
weights = np.array([0.1, 0.85, 0.05])

# Solve the system of equations with bounds to avoid negative sqrt issues
result = least_squares(equations, initial_guesses, bounds=(0, np.inf))

# Extract the solutions
x_values = result.x

# Print the solutions
for i, x in enumerate(x_values, 1):
    print(f"x{i} = {x}")

# Calculate and print the values of each mu_i + sqrt(((m+1)*ln t)/x_i)
for i in range(len(x_values)):
    print(f"ucb {i}: {mu_values[i] + np.sqrt(((m + 1) * np.log(t)) / x_values[i])}")

print(f"sum obs: { (weights @ x_values) * n}, t*m*n={t*m}")