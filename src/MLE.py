# Maximum likelihood estimation with intrinsic scatter as an additional
# parameter. The optimized parameters become the starting point for MCMC.
# ===========================================
# 7. Define the likelihood function
# Include model uncertainty through intrinsic scatter.
# ===========================================

def log_likelihood(theta, x, y, yerr):
    """
    Log-likelihood function.

    Parameters:
    theta = [m, b, log_f]
      m: slope, equal to 1/H0
      b: intercept
      log_f: logarithm of the intrinsic-scatter factor
    x: velocity data
    y: distance data
    yerr: distance uncertainty

    Returns:
    log-likelihood value
    """
    m, b, log_f = theta

    # Linear model: d = m*v + b.
    model = m * x + b

    # Total variance = measurement error^2 + (model * exp(log_f))^2.
    # exp(log_f) is the intrinsic-scatter scale factor.
    sigma2 = yerr**2 + model**2 * np.exp(2 * log_f)

    # Gaussian log-likelihood.
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

# ===========================================
# 8. Maximum likelihood estimation with scipy
# ===========================================

from scipy.optimize import minimize

# Set a random seed for reproducibility.
np.random.seed(42)

# Define the negative log-likelihood for minimization.
nll = lambda *args: -log_likelihood(*args)

# Initial guesses:
# m ~= 1/72 ~= 0.0139, corresponding to H0 ~= 72 km/s/Mpc
# b ~= 0
# log_f ~= 0, corresponding to f ~= 1
initial = np.array([1 / 72.0, 0.0, np.log(1.0)]) + 0.1 * np.random.randn(3)

# Use numerical optimization to find the maximum-likelihood estimate.
soln = minimize(nll, initial, args=(x_new, y_new, yerr_new))
m_ml, b_ml, log_f_ml = soln.x

print("=" * 50)
print("Maximum likelihood result:")
print("=" * 50)
print("Slope m, equal to 1/H0 = {0:.6f}".format(m_ml))
print("Intercept b = {0:.3f}".format(b_ml))
print("Intrinsic-scatter factor f = {0:.3f}".format(np.exp(log_f_ml)))
print("Hubble constant H0 = 1/m = {0:.2f} km/s/Mpc".format(1/m_ml))

# Visualize the maximum-likelihood fit.
plt.figure(figsize=(10, 6))
plt.errorbar(x_new, y_new, yerr=yerr_new,
             fmt=".k", capsize=0, alpha=0.5)
plt.plot(x0, np.dot(np.vander(x0, 2), w), "--k",
         label="Least Squares", linewidth=2)
plt.plot(x0, np.dot(np.vander(x0, 2), [m_ml, b_ml]), ":k",
         label="Maximum Likelihood", linewidth=3)
plt.ylabel("Distance [Mpc]")
plt.xlabel("Velocity [km/s]")
plt.title("Least squares vs. maximum likelihood fitting")
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()
