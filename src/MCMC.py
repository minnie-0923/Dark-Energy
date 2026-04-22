# 这个方法用到的核心思想是贝叶斯定理：后验=先验✖️似然
# 使用的随机游走的起点是用最大似然估计得出来的结果
# 最后的结果进行了收敛性验证，同时对数据也进行了老化处理和稀疏处理
# 老化处理：把最开始的一部分游走删去；稀疏处理：每15步取一个样本
# 在最后结果中对于三个参数：m,b,log(f)进行相关性分析，得到最后的corner图，在图中发现m,b有相关性，和物理推测相符
# m和log(f),b和log(f)都是没有相关性的
# ===========================================
# 9. 定义先验分布（贝叶斯分析）
# ===========================================

def log_prior(theta):
    """
    对数先验函数

    参数：
    theta = [m, b, log_f]

    返回：
    先验概率的对数（无信息先验）
    """
    m, b, log_f = theta

    # 设定参数的合理范围：
    # m: 0.0到0.5 (对应H₀从2到无穷大)
    # b: -100到100 Mpc
    # log_f: -10到1 (对应f从4.5e-5到2.7)

    if 0.0 < m < 0.5 and -100.0 < b < 100.0 and -10.0 < log_f < 1.0:
        return 0.0  # 在范围内返回常数（均匀先验）
    return -np.inf  # 在范围外返回负无穷（概率为0）

def log_probability(theta, x, y, yerr):
    """
    对数后验概率（贝叶斯定理）

    后验 ∝ 先验 × 似然
    在对数空间中：log(后验) = log(先验) + log(似然)
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)
# ===========================================
# 10. 安装并使用emcee进行MCMC采样
# ===========================================

# 安装emcee库（在Colab中）
!pip install -U emcee

import emcee

# 设置MCMC的初始位置（以最大似然解为中心）
pos = soln.x + 1e-4 * np.random.randn(32, 3)  # 32个行走者，每个有3个参数
nwalkers, ndim = pos.shape

print("=" * 50)
print("开始MCMC采样...")
print("行走者数量:", nwalkers)
print("参数维度:", ndim)
print("=" * 50)

# 创建EnsembleSampler对象
sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(x_new, y_new, yerr_new)
)

# 运行MCMC采样（5000步）
sampler.run_mcmc(pos, 5000, progress=True)

print("\nMCMC采样完成！")
print("总步数:", 5000)
print("总样本数:", nwalkers * 5000)

# ===========================================
# 11. 检查MCMC链的收敛性
# ===========================================

# 绘制参数链的轨迹图
fig, axes = plt.subplots(3, figsize=(12, 8), sharex=True)
samples = sampler.get_chain()  # 获取所有链
labels = ["m (1/H₀)", "b [Mpc]", "log(f)"]

for i in range(ndim):
    ax = axes[i]
    # 绘制每个行走者的链
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

# 计算自相关时间（评估链的收敛性）
tau = sampler.get_autocorr_time()
print("=" * 50)
print("自相关时间（收敛性指标）：")
print("=" * 50)
for i, label in enumerate(labels):
    print(f"{label}: {tau[i]:.1f} 步")
print("\n注：自相关时间越小，采样效率越高")
# ===========================================
# 12. 处理MCMC样本（去除老化期，稀释样本）
# ===========================================
# 去除前100步作为老化期（burn-in），每15步取一个样本（稀释）
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
print(f"\n处理后样本形状: {flat_samples.shape}")
print(f"有效样本数: {flat_samples.shape[0]}")
# ===========================================
# 13. 绘制后验分布的相关图（corner plot）
# ===========================================

# 安装corner库
!pip install corner

import corner

fig = corner.corner(
    flat_samples,
    labels=[r"$m = 1/H_0$ [Mpc·s/km]",
            r"$b$ [Mpc]",
            r"$\log\,f$"],
    quantiles=[0.16, 0.5, 0.84],  # 显示16%, 50%, 84%分位数
    show_titles=True,
    title_kwargs={"fontsize": 12},
    label_kwargs={"fontsize": 14}
)

# 在图上标注哈勃常数值
H0_samples = 1 / flat_samples[:, 0]  # 计算H₀ = 1/m
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
# 14. 可视化MCMC拟合的不确定性
# ===========================================

plt.figure(figsize=(12, 8))

# 从后验分布中随机抽取100个样本并绘制对应的拟合线
inds = np.random.randint(len(flat_samples), size=100)
for ind in inds:
    sample = flat_samples[ind]
    plt.plot(x0, np.dot(np.vander(x0, 2), sample[:2]),
             "C1", alpha=0.05, linewidth=1)

# 绘制数据点
plt.errorbar(x_new, y_new, yerr=yerr_new,
             fmt=".k", capsize=0, alpha=0.5, label="Observational data")

# 绘制最小二乘拟合
plt.plot(x0, np.dot(np.vander(x0, 2), w), "--k",
         label="Least Squares", linewidth=2)

# 绘制最大似然拟合
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
# 15. 输出最终结果
# ===========================================

from IPython.display import display, Math

print("=" * 60)
print("哈勃常数测量最终结果（贝叶斯MCMC分析）")
print("=" * 60)
print()

# 对每个参数计算中位数和68%置信区间
for i in range(ndim):
    # 计算16%, 50%, 84%分位数
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)  # 计算上下误差

    # 创建LaTeX格式的输出
    if i == 0:  # 对于斜率m，额外计算H₀
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
print("分析完成！")
print("=" * 60)
