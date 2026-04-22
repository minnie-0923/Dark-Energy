# 通过可视化的方法，简单分析数据分布

# ===========================================
# 2. 数据可视化（对数坐标）
# 是因为线性坐标 近的地方数据比较密集 远的地方数据比较稀疏
# 拟合的时候可能会对近的拟合权重高 远的地方就考虑不到
# ===========================================

plt.figure(figsize=(10, 6))
# 绘制速度-距离关系的散点图（对数坐标）
plt.plot(distance.to(u.mpc).value, velocity.to(u.km / u.s).value,
         marker=".", color="black", linestyle="none")
plt.xscale("log")  # x轴使用对数坐标
plt.yscale("log")  # y轴使用对数坐标
plt.xlabel("Distance [Mpc]")
plt.ylabel("Velocity [km/s]")
plt.title("Hubble plot: Galaxy velocity vs. distance (logarithmic coordinates)")
plt.grid(True, alpha=0.3)
plt.show()



# ===========================================
# 3. 数据可视化（线性坐标，带误差条）
# 为了和对数坐标对比
# ===========================================

plt.figure(figsize=(10, 6))
# 绘制带误差条的散点图
plt.errorbar(distance.to(u.mpc).value, velocity.to(u.km / u.s).value,
             xerr=distance_error.to(u.mpc).value,  # 距离误差
             marker=".", color="black", linestyle="none",
             ecolor='red', alpha=0.5, capsize=2)
plt.xlim(0, 700)  # 限制x轴范围
plt.ylim(0, 4e4)  # 限制y轴范围
plt.xlabel("Distance [Mpc]")
plt.ylabel("Velocity [km/s]")
plt.title("Hubble plot: Galaxy velocity vs. distance (linear coordinates, with error)")
plt.grid(True, alpha=0.3)
plt.show()
