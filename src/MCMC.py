# The core idea is Bayes' theorem: posterior = prior * likelihood.
# The walkers start near the maximum-likelihood estimate.
# The final chain is checked for convergence, then burn-in removal and
# thinning are applied. The corner plot analyzes correlations among
# m, b, and log(f); the slope and intercept are correlated, while log(f)
# is largely independent of the other two parameters.
# ===========================================
# 9. Define the prior distribution for Bayesian analysis
# ===========================================

def log_prior(theta):
    """
    Log-prior function.

    Parameters:
    theta = [m, b, log_f]

    Returns:
    log prior probability under a weakly informative uniform prior
    """
    m, b, log_f = theta

    # Physically reasonable parameter ranges:
    # m: 0.0 to 0.5, corresponding to H0 from 2 to infinity
    # b: -100 to 100 Mpc
    # log_f: -10 to 1, corresponding to f from 4.5e-5 to 2.7

    if 0.0 < m < 0.5 and -100.0 < b < 100.0 and -10.0 < log_f < 1.0:
        return 0.0  # constant value inside the prior range
    return -np.inf  # zero probability outside the range

def log_probability(theta, x, y, yerr):
    """
    Log posterior probability from Bayes' theorem.

    posterior is proportional to prior times likelihood.
    In log space: log posterior = log prior + log likelihood.
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)
# ===========================================
# 10. Install and use emcee for MCMC sampling
# ===========================================

# Install emcee in Colab.
!pip install -U emcee

import emcee

# Set MCMC initial positions around the maximum-likelihood solution.
pos = soln.x + 1e-4 * np.random.randn(32, 3)  # 32 walkers, 3 parameters each
nwalkers, ndim = pos.shape

print("=" * 50)
print("Starting MCMC sampling...")
print("Number of walkers:", nwalkers)
print("Parameter dimensions:", ndim)
print("=" * 50)

# Create the EnsembleSampler object.
sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(x_new, y_new, yerr_new)
)

# Run MCMC sampling for 5,000 steps.
sampler.run_mcmc(pos, 5000, progress=True)

print("\nMCMC sampling completed.")
print("Total steps:", 5000)
print("Total samples:", nwalkers * 5000)

# ===========================================
# 11. Check MCMC convergence
# ===========================================

# Plot parameter-chain traces.
fig, axes = plt.subplots(3, figsize=(12, 8), sharex=True)
samples = sampler.get_chain()  # all chains
labels = ["m (1/H₀)", "b [Mpc]", "log(f)"]

for i in range(ndim):
    ax = axes[i]
    # Plot each walker chain.
    for j in range(nwalkers):
        ax.plot(samples[:, j, i], alpha=0.3, linewidth=0.5)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i], fontsize=12)
    ax.yaxis.set_label_coords(-0.1, 0.5)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Number of iterations", fontsize=12)
plt.suptitle("MCMC parameter chain trajectory diagram", fontsize=14)
plt.tight_layout()
plt.show()

# Compute autocorrelation time as a convergence diagnostic.
tau = sampler.get_autocorr_time()
print("=" * 50)
print("Autocorrelation time, a convergence diagnostic:")
print("=" * 50)
for i, label in enumerate(labels):
    print(f"{label}: {tau[i]:.1f} steps")
print("\nNote: shorter autocorrelation time indicates more efficient sampling.")
# ===========================================
# 12. Process MCMC samples with burn-in removal and thinning
# ===========================================
# Remove the first 100 steps as burn-in, then keep every 15th sample.
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
print(f"\nProcessed sample shape: {flat_samples.shape}")
print(f"Effective sample count: {flat_samples.shape[0]}")
# ===========================================
# 13. Plot posterior correlations with a corner plot
# ===========================================

# Install corner.
!pip install corner

import corner

fig = corner.corner(
    flat_samples,
    labels=[r"$m = 1/H_0$ [Mpc·s/km]",
            r"$b$ [Mpc]",
            r"$\log\,f$"],
    quantiles=[0.16, 0.5, 0.84],  # show 16%, 50%, and 84% quantiles
    show_titles=True,
    title_kwargs={"fontsize": 12},
    label_kwargs={"fontsize": 14}
)

# Annotate the Hubble constant value.
H0_samples = 1 / flat_samples[:, 0]  # H0 = 1/m
H0_median = np.median(H0_samples)
H0_lower = H0_median - np.percentile(H0_samples, 16)
H0_upper = np.percentile(H0_samples, 84) - H0_median

fig.text(0.5, 0.95,
         r"$H_0 = {:.1f}^{{+{:.1f}}}_{{-{:.1f}}}$ km/s/Mpc".format(
             H0_median, H0_upper, H0_lower),
         ha='center', fontsize=14)
plt.suptitle("Posterior distribution correlation plot", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# ===========================================
# 14. Visualize MCMC fitting uncertainty
# ===========================================

plt.figure(figsize=(12, 8))

# Draw 100 posterior samples and plot their corresponding fit lines.
inds = np.random.randint(len(flat_samples), size=100)
for ind in inds:
    sample = flat_samples[ind]
    plt.plot(x0, np.dot(np.vander(x0, 2), sample[:2]),
             "C1", alpha=0.05, linewidth=1)

# Plot observed data.
plt.errorbar(x_new, y_new, yerr=yerr_new,
             fmt=".k", capsize=0, alpha=0.5, label="Observational data")

# Plot least-squares fit.
plt.plot(x0, np.dot(np.vander(x0, 2), w), "--k",
         label="Least Squares", linewidth=2)

# Plot maximum-likelihood fit.
plt.plot(x0, np.dot(np.vander(x0, 2), [m_ml, b_ml]), ":k",
         label="Maximum Likelihood", linewidth=2)

plt.ylabel("Distance [Mpc]", fontsize=14)
plt.xlabel("Velocity [km/s]", fontsize=14)
plt.title("MCMC fitting uncertainty (100 posterior samples)", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ===========================================
# 15. Output final result
# ===========================================

from IPython.display import display, Math

print("=" * 60)
print("Final Hubble constant measurement from Bayesian MCMC analysis")
print("=" * 60)
print()

# Compute the median and 68% credible interval for each parameter.
for i in range(ndim):
    # Compute the 16%, 50%, and 84% quantiles.
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)  # lower and upper uncertainties

    # Create LaTeX-formatted output.
    if i == 0:  # for slope m, also compute H0
        H0_median = 1 / mcmc[1]
        H0_upper = 1 / (mcmc[1] - q[0]) - H0_median
        H0_lower = H0_median - 1 / (mcmc[1] + q[1])

        txt1 = r"m = 1/H_0 = {0:.6f}_{{-{1:.6f}}}^{{+{2:.6f}}}".format(
            mcmc[1], q[0], q[1])
        txt2 = r"H_0 = {0:.2f}_{{-{1:.2f}}}^{{+{2:.2f}}} \ \mathrm{{km/s/Mpc}}".format(
            H0_median, H0_lower, H0_upper)

        display(Math(txt1))
        display(Math(txt2))
        print()
    else:
        txt = r"\mathrm{{{3}}} = {0:.5f}_{{-{1:.5f}}}^{{+{2:.5f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        display(Math(txt))
        print()

print("=" * 60)
print("Analysis complete.")
print("=" * 60)
