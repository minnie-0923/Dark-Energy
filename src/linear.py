# Two linear fits are used: ordinary least squares and weighted least
# squares, where inverse variance represents measurement precision.
# ===========================================
# 4. Simple linear fitting, ordinary least squares
# ===========================================

# Extract data values as numpy arrays.
x = distance.to(u.mpc).value  # distance in Mpc
y = velocity.to(u.km / u.s).value  # velocity in km/s

# Select points in the 0-700 Mpc range.
ind = np.where((x > 0) & (x < 700))

# Use numpy.polyfit for the direct relation v = H0 * d.
# z = [slope H0, intercept]
z = np.polyfit(x[ind], y[ind], 1)

# Create a polynomial object for later evaluation.
p = np.poly1d(z)

# Compute model predictions.
velocity_model = p(x[ind])

# Visualize the fit.
plt.figure(figsize=(10, 6))
plt.errorbar(x[ind], y[ind],
             xerr=distance_error.to(u.mpc).value[ind],
             marker=".", color="black", linestyle="none",
             ecolor='red', alpha=0.5, capsize=2)
plt.plot(x[ind], velocity_model, color="grey", linewidth=5, alpha=0.7)
plt.xlim(0, 700)
plt.ylim(0, 4e4)
plt.xlabel("Distance [Mpc]")
plt.ylabel("Velocity [km/s]")
plt.title("Simple linear fit: v = H₀ × d")
plt.grid(True, alpha=0.3)

# Annotate the Hubble constant estimate.
plt.annotate(r"$H_0$ = {0:5.2f} km/s/Mpc".format(z[0]),
             xy=(50, 30000), fontsize=14)
plt.show()

# ===========================================
# 6. Matrix-form weighted least squares
# The weights encode measurement precision.
# ===========================================

# Create a Vandermonde matrix for fitting d = m*v + b.
A = np.vander(x_new, 2)

# Create a diagonal covariance matrix from squared uncertainties.
C = np.diag(yerr_new**2)

# Compute the weighted least-squares solution:
# w = (A^T W A)^-1 A^T W y, where W = C^-1.
ATA = np.dot(A.T, A / (yerr_new**2)[:, None])
cov = np.linalg.inv(ATA)  # parameter covariance matrix
w = np.linalg.solve(ATA, np.dot(A.T, y_new / yerr_new**2))

print("=" * 50)
print("Weighted least-squares result:")
print("=" * 50)
print("Slope m, equal to 1/H0 = {0:.6f} +/- {1:.6f}".format(w[0], np.sqrt(cov[0, 0])))
print("Intercept b = {0:.3f} +/- {1:.3f}".format(w[1], np.sqrt(cov[1, 1])))
print("Hubble constant H0 = 1/m = {0:.2f} +/- {1:.2f} km/s/Mpc".format(
    1/w[0], np.sqrt(cov[0, 0])/w[0]**2))
print()

# Visualize the weighted least-squares fit.
plt.figure(figsize=(10, 6))
plt.errorbar(x_new, y_new, yerr=yerr_new,
             fmt=".k", capsize=0, alpha=0.5)
x0 = np.linspace(0, 50000, 500)  # velocity values for the fit line
plt.plot(x0, np.dot(np.vander(x0, 2), w), "--k",
         label="Least squares fitting", linewidth=2)
plt.ylabel("Distance [Mpc]")
plt.xlabel("Velocity [km/s]")
plt.title("Weighted least squares fitting")
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()
