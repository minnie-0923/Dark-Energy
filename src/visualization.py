# Visual exploration of the data distribution.

# ===========================================
# 2. Data visualization, logarithmic axes
# Log scaling helps compare nearby dense data and distant sparse data
# without letting the nearby points visually dominate the relation.
# ===========================================

plt.figure(figsize=(10, 6))
# Plot the velocity-distance relation on logarithmic axes.
plt.plot(distance.to(u.mpc).value, velocity.to(u.km / u.s).value,
         marker=".", color="black", linestyle="none")
plt.xscale("log")  # logarithmic x-axis
plt.yscale("log")  # logarithmic y-axis
plt.xlabel("Distance [Mpc]")
plt.ylabel("Velocity [km/s]")
plt.title("Hubble plot: Galaxy velocity vs. distance (logarithmic coordinates)")
plt.grid(True, alpha=0.3)
plt.show()



# ===========================================
# 3. Data visualization, linear axes with error bars
# This provides a direct comparison with the logarithmic view.
# ===========================================

plt.figure(figsize=(10, 6))
# Plot the scatter diagram with distance error bars.
plt.errorbar(distance.to(u.mpc).value, velocity.to(u.km / u.s).value,
             xerr=distance_error.to(u.mpc).value,  # distance uncertainty
             marker=".", color="black", linestyle="none",
             ecolor='red', alpha=0.5, capsize=2)
plt.xlim(0, 700)  # x-axis range
plt.ylim(0, 4e4)  # y-axis range
plt.xlabel("Distance [Mpc]")
plt.ylabel("Velocity [km/s]")
plt.title("Hubble plot: Galaxy velocity vs. distance (linear coordinates, with error)")
plt.grid(True, alpha=0.3)
plt.show()
