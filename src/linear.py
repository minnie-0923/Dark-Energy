# 使用两种线性拟合：第一种是简单的最小二乘法；第二种使用加权重的最小二乘法，权重是用误差的倒数表示
# ===========================================
# 4. 简单线性拟合（最小二乘法）
# ===========================================

# 提取数据值（转换为numpy数组）
x = distance.to(u.mpc).value  # 距离（Mpc）
y = velocity.to(u.km / u.s).value  # 速度（km/s）

# 选择距离在0-700 Mpc范围内的数据点
ind = np.where((x > 0) & (x < 700))

# 使用numpy的polyfit进行线性拟合（v = H₀ * d）
# 返回的z = [斜率H₀, 截距]
z = np.polyfit(x[ind], y[ind], 1)

# 创建多项式函数对象（方便后续计算）
p = np.poly1d(z)

# 计算模型预测值
velocity_model = p(x[ind])

# 可视化拟合结果
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

# 在图上标注哈勃常数估计值
plt.annotate(r"$H_0$ = {0:5.2f} km/s/Mpc".format(z[0]),
             xy=(50, 30000), fontsize=14)
plt.show()

# ===========================================
# 6. 矩阵形式的加权最小二乘法
# 这是传统的最小二乘回归方法
# 权重就是精确度
# ===========================================

# 创建Vandermonde矩阵（用于多项式拟合）
A = np.vander(x_new, 2)  # 生成[1, x]矩阵，用于拟合 d = m*v + b

# 创建协方差矩阵（对角矩阵，对角线元素为误差平方）
C = np.diag(yerr_new**2)

# 计算加权最小二乘的解：w = (AᵀWA)⁻¹ AᵀWy
# 其中W = C⁻¹
ATA = np.dot(A.T, A / (yerr_new**2)[:, None])
cov = np.linalg.inv(ATA)  # 参数协方差矩阵
w = np.linalg.solve(ATA, np.dot(A.T, y_new / yerr_new**2))

print("=" * 50)
print("加权最小二乘法结果：")
print("=" * 50)
print("斜率 m (即 1/H₀) = {0:.6f} ± {1:.6f}".format(w[0], np.sqrt(cov[0, 0])))
print("截距 b = {0:.3f} ± {1:.3f}".format(w[1], np.sqrt(cov[1, 1])))
print("哈勃常数 H₀ = 1/m = {0:.2f} ± {1:.2f} km/s/Mpc".format(
    1/w[0], np.sqrt(cov[0, 0])/w[0]**2))
print()

# 可视化最小二乘拟合结果
plt.figure(figsize=(10, 6))
plt.errorbar(x_new, y_new, yerr=yerr_new,
             fmt=".k", capsize=0, alpha=0.5)
x0 = np.linspace(0, 50000, 500)  # 生成用于绘制拟合线的速度值
plt.plot(x0, np.dot(np.vander(x0, 2), w), "--k",
         label="Least squares fitting", linewidth=2)
plt.ylabel("Distance [Mpc]")
plt.xlabel("Velocity [km/s]")
plt.title("Weighted least squares fitting")
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()
