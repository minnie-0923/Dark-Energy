# 简单的最大似然估计，把固有弥散（系统误差）作为参数考虑进来，同时也使用了高斯似然函数，得到的结果作为下一步MCMC游走的起始点
# ===========================================
# 7. 定义似然函数（最大似然估计）
# 考虑模型的不确定性（固有弥散）
# ===========================================

def log_likelihood(theta, x, y, yerr):
    """
    对数似然函数

    参数：
    theta = [m, b, log_f] - 拟合参数
      m: 斜率 (1/H₀)
      b: 截距
      log_f: 固有弥散的对数
    x: 速度数据
    y: 距离数据
    yerr: 距离测量误差

    返回：
    对数似然值
    """
    m, b, log_f = theta

    # 线性模型：d = m*v + b
    model = m * x + b

    # 总方差 = 测量误差² + (模型 × exp(log_f))²
    # exp(log_f) 是固有弥散的比例因子
    sigma2 = yerr**2 + model**2 * np.exp(2 * log_f)

    # 高斯似然函数：-0.5 * Σ[(y-model)²/sigma² + log(sigma²)]
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

# ===========================================
# 8. 最大似然估计（使用scipy优化）
# ===========================================

from scipy.optimize import minimize

# 设置随机种子以保证结果可重复
np.random.seed(42)

# 定义负对数似然函数（用于最小化）
nll = lambda *args: -log_likelihood(*args)

# 初始猜测值：
# m ≈ 1/72 ≈ 0.0139 (对应H₀≈72 km/s/Mpc)
# b ≈ 0
# log_f ≈ 0 (对应f≈1)
initial = np.array([1 / 72.0, 0.0, np.log(1.0)]) + 0.1 * np.random.randn(3)

# 使用数值优化找到最大似然估计
soln = minimize(nll, initial, args=(x_new, y_new, yerr_new))
m_ml, b_ml, log_f_ml = soln.x

print("=" * 50)
print("最大似然估计结果：")
print("=" * 50)
print("斜率 m (即 1/H₀) = {0:.6f}".format(m_ml))
print("截距 b = {0:.3f}".format(b_ml))
print("固有弥散因子 f = {0:.3f}".format(np.exp(log_f_ml)))
print("哈勃常数 H₀ = 1/m = {0:.2f} km/s/Mpc".format(1/m_ml))

# 可视化最大似然拟合结果
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
