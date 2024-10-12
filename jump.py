import numpy as np
import matplotlib.pyplot as plt

def simulate_jump_diffusion(S0, mu, sigma, T, dt, lamb, mu_J, sigma_J, num_simulations):
    # Number of time steps
    N = int(T / dt)
    
    # Initialize the array to store the simulated stock prices
    stock_paths = np.zeros((num_simulations, N))
    stock_paths[:, 0] = S0
    
    for sim in range(num_simulations):
        for t in range(1, N):
            # Generate the Brownian motion part
            dW = np.random.normal(0, np.sqrt(dt))
            
            # Generate the Poisson jump part
            J = np.random.poisson(lamb * dt)
            Y = np.random.normal(mu_J, sigma_J, J)
            
            # Calculate the jump component
            jump = np.sum(np.exp(Y) - 1)
            
            # Update the stock price
            stock_paths[sim, t] = stock_paths[sim, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW) * (1 + jump)
    
    return stock_paths

# Parameters
S0 = 100  # Initial stock price
mu = 0.05  # Drift
sigma = 0.2  # Volatility
T = 1  # Time horizon in years
dt = 1/252  # Time step (daily)
lamb = 1  # Average number of jumps per year
mu_J = 0.02  # Mean of jump size
sigma_J = 0.1  # Standard deviation of jump size
num_simulations = 10  # Number of simulations

# Simulate the stock price paths
stock_paths = simulate_jump_diffusion(S0, mu, sigma, T, dt, lamb, mu_J, sigma_J, num_simulations)

# Plot the simulated stock price paths
plt.figure(figsize=(10, 6))
for i in range(num_simulations):
    plt.plot(stock_paths[i, :], lw=0.8)
plt.title('Jump-Diffusion Model - Simulated Stock Price Paths')
plt.xlabel('Time (days)')
plt.ylabel('Stock Price')
plt.show()