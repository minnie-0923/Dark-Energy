# Dark-Energy
learn something about astrophysics!

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)


一个用于分析 Tonry 2003 超新星数据的 Python 项目，通过线性回归、最大似然估计（MLE）和贝叶斯 MCMC 方法精确测量哈勃常数 $H_0$。

## 项目概述

本项目利用 Ia 型超新星的观测数据来验证哈勃定律，并估算哈勃常数 $H_0$：

- 数据集: 使用 Tonry 2003 公布的红移-距离模数数据集。
- 物理模型: 哈勃定律 $v = H_0 \times d$。
- 核心目标: 获得 $H_0$ 的最优估计及其不确定性。





## 项目结构

```bash
Dark-Energy/
├── data/                    # 原始天文数据
│   └── Tonry_2003.vot       # Tonry 2003 观测表格
├── src/                     # 源代码
│   ├── data.py              # 数据加载及预处理
│   ├── visualization.py     # 数据可视化
│   ├── linear.py            # 线性拟合
│   ├── MLE.py               # 最大似然估计
│   └── MCMC.py              # MCMC采样
├── notebooks/               # Jupyter 笔记本
│   └── darkenergy.ipynb
├── results/                 # 分析结果
│   ├── plots/               # 数据预处理图
│   ├── linear/              # 线性拟合图
│   ├── MLE/                 # 最大似然估计图
│   └── MCMC/                # MCMC采样图
├── requirements.txt
└── README.md
```

## 快速开始

### 安装步骤

1. 克隆仓库

   ```bash
   git clone https://github.com/minnie-0923/Dark-Energy.git
   cd Dark-Energy
   ```

2. 安装依赖

   ```bash
   pip install -r requirement.txt
   ```

### 基本使用

运行主分析笔记本：

```python
# 打开Jupyter笔记本
jupyter notebook notebooks/darkenergy.ipynb
```

## 分析流程

### 1. 数据预处理 (Data Preprocessing)

- 格式解析: 使用 `astropy.io.ascii` 读取 VOTable 数据。
- 量纲转换: 
  - 将距离模数（Distance Modulus）转换为线性距离（Mpc）。
  - 处理速度单位（km/s）。
- 误差传播: 考虑到距离模数是对数尺度，将其误差转换为线性空间下的非对称误差。

### 2. 可视化分析 (Visualization)

- 对数坐标分布: 通过对比线性和对数的可视化方式，选择对数可视化，解决近处数据密集、远处数据稀疏的拟合权重倾斜问题。
- 误差条图: 在线性坐标下展示速度与距离的关系及其观测不确定性，远处不确定性更大，近处不确定性更小，为后面的带权重最小二乘法做数据指导。
- 数据筛选: 选取距离在 0-700 Mpc 范围内的良好样本进行拟合。

### 3. 线性拟合 (Linear Fitting)

- 坐标变换: 考虑到距离测量误差远大于速度测量误差，将距离设为因变量 $d = m \cdot v + b$。
- 加权最小二乘 (WLS):使用误差的倒数作为准确度权重
  - 构建协方差矩阵 $C$。
  - 通过矩阵运算 $\mathbf{w} = (A^T C^{-1} A)^{-1} A^T C^{-1} \mathbf{y}$ 获取解析解。
  - 估算初步的哈勃常数 $H_0 = 1/m$。
### 4. 最大似然估计 (MLE)

- 似然函数定义: 考虑到系统的固有误差难以衡量，引入固有弥散因子 $f$，同时鉴于数据的自然分布性构建高斯似然函数。
- 异方差建模: 总方差 $\sigma^2 = \sigma_{\text{err}}^2 + \text{model}^2 \cdot \exp(2\ln f)$。
- 数值优化: 使用 `scipy.optimize.minimize` 寻找参数空间的最优解，作为 MCMC 的初始位置。

### 5. MCMC 采样 (MCMC Sampling)

- 贝叶斯框架: 
  - 先验 (Prior): 设定参数的物理合理范围（无信息先验）。
  - 后验 (Posterior): $\text{log-post} = \text{log-prior} + \text{log-likelihood}$。
- 采样策略: 
  - 使用 `emcee` 的仿射不变集成采样器。
  - 32 个行走者进行 5000 次迭代。
  - 对数据进行老化处理和稀疏处理。
- 结果导出:
  - 绘制参数演化迹图，检查自相关时间确保收敛。
  - 绘制 Corner Plot 展示 $H_0$ 与其它参数的关联。
  - 发现m和b有相关性，两者均与log(f)无相关性。



## 技术细节

### 主要依赖库

- Astropy: 天文坐标与单位计算。
- Emcee: 仿射不变 MCMC 采样器。
- Corner: 后验分布可视化。
- Scipy: 数值优化与矩阵运算。

### 1. 贝叶斯框架

在贝叶斯统计中，参数的后验概率分布与似然函数和先验分布的乘积成正比：
$$P(\theta \mid \text{data}) \propto P(\text{data} \mid \theta) \times P(\theta)$$
其中：
- $P(\text{data} \mid \theta)$ 是似然函数，描述了在给定参数 $\theta$ 下观测到当前数据的概率；
- $P(\theta)$ 是先验分布，代表我们在观测数据之前对参数的信念或假设。

在项目中，我们对数空间进行计算以提高数值稳定性，即：
$$\log P(\theta \mid \text{data}) = \log P(\text{data} \mid \theta) + \log P(\theta)$$

对应的核心代码如下：

```python
def log_probability(theta, x, y, yerr):
    """
    对数后验概率（贝叶斯定理）
    log(后验) = log(先验) + log(似然)
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)
```


### 2. 似然函数

项目采用高斯似然函数，但与传统最小二乘不同，它额外引入了一个固有弥散因子 $f$，用于描述模型本身无法解释的额外散布（例如 Ia 型超新星的内禀光度离散）。

模型预测为：
$$\text{model} = m \cdot v + b$$
总方差由测量误差和模型相对弥散共同构成：
$$\sigma^2 = \sigma_{\text{meas}}^2 + (\text{model} \cdot f)^2$$
其中 $f = \exp(\log f)$，而 $\log f$ 是待拟合参数之一。

对数似然函数的形式为：
$$\log \mathcal{L} = -\frac{1}{2} \sum_i \left[ \frac{(y_i - \text{model}_i)^2}{\sigma_i^2} + \log(\sigma_i^2) \right]$$

代码实现如下：

```python
def log_likelihood(theta, x, y, yerr):
    m, b, log_f = theta
    model = m * x + b
    sigma2 = yerr2 + model2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - model)  2 / sigma2 + np.log(sigma2))
```

### 3. MCMC 游走

项目使用 Python 库 `emcee` 执行马尔可夫链蒙特卡罗（MCMC）采样，以获得参数的后验分布样本。

游走过程：
1. 以最大似然估计（MLE）的结果为中心，添加微小扰动，生成 32 个“行走者”（walkers）的初始位置。
2. 创建 `EnsembleSampler` 对象，传入对数概率函数 `log_probability` 及数据。
3. 运行 5000 步 MCMC，每个 walker 产生一条链，最终得到 $32 \times 5000 = 160,000$ 个后验样本。
4. 对于后验样本进行老化处理和稀疏处理。

核心代码如下：

```python
pos = soln.x + 1e-4 * np.random.randn(32, 3)   # soln.x 是最大似然拟合结果
nwalkers, ndim = pos.shape

# 创建采样器
sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(x_new, y_new, yerr_new)
)

# 运行 MCMC 采样（5000步）
sampler.run_mcmc(pos, 5000, progress=True)
```

运行后，可通过 `sampler.chain` 获取所有链的样本，用于后续的参数推断和不确定性估计。

## 结果示例
项目生成的分析结果包括：

- 原始数据散点图：用线性和对数的形式展示原始数据。
- 哈勃图: 包含 WLS、MLE 与 MCMC 后验线条的对比。
- 参数轨迹图: 用于诊断 MCMC 链的收敛状态。
- Corner Plot: m, b, $log_f$分别对应$H_0$、截距与弥散因子,展示了他们的相关性。


## 物理背景

### 暗能量与哈勃常数 $H_0$

现代宇宙学建立在广义相对论与宇宙学原理之上。20 世纪末，通过 Ia 型超新星的观测发现宇宙正在加速膨胀，这一颠覆性发现直接指向一种未知的能量组分——暗能量 (Dark Energy)。在标准 $\Lambda$CDM 模型中，暗能量约占宇宙总能量密度的 $68\%$，其性质由状态方程参数 $w = p / \rho$ 描述（对宇宙学常数 $\Lambda$ 有 $w = -1$）。

宇宙膨胀的动力学由弗里德曼方程支配。在低红移 ($z \ll 1$) 极限下，该方程退化为经典的哈勃定律：

$$ v = H_0 \times d $$

其中：
- $v$ 为星系因宇宙膨胀而产生的退行速度 (km/s)；
- $d$ 为星系与观测者之间的物理距离 (Mpc)；
- $H_0$ 为当前宇宙的哈勃常数，量纲为 $\text{km} \cdot \text{s}^{-1} \cdot \text{Mpc}^{-1}$。

$H_0$ 是刻画宇宙膨胀速率的最核心参数，它不仅决定了宇宙的年龄与尺度，更是连接早期宇宙（微波背景辐射）与晚期宇宙（超新星）观测的桥梁。当前关于 $H_0$ 的测量存在显著的 “哈勃张力” (Hubble Tension)：普朗克卫星给出的值为 $H_0 \approx 67.4$，而基于超新星的局域距离阶梯测量结果为 $H_0 \approx 73.0$。这一差异可能暗示着超出标准模型的新物理，因此通过独立、纯净的超新星样本精确测定低红移下的 $H_0$ 对于理解暗能量本质和检验宇宙学模型至关重要。

### Ia 型超新星

Ia 型超新星 (Type Ia Supernovae, SNe Ia) 是碳氧白矮星在接近钱德拉塞卡极限 ($\sim 1.4 M_\odot$) 时发生的剧烈热核爆炸。这类天体的物理过程具有高度的一致性：

1. 标准烛光特性：尽管 Ia 超新星的本征光度并非完全一致，但通过光变曲线形状与颜色的经验校正（如 Phillips 关系），其峰值绝对星等的弥散可被压缩至 $\sim 0.15$ 等，从而成为宇宙学中极为精准的标准烛光。
2. 距离模数测量：通过比较观测到的视星等 $m$ 与校正后的绝对星等 $M$，可获得距离模数 $\mu = m - M = 5 \log_{10}(d_L / 10 \text{ pc})$，进而导出光度距离 $d_L$。在低红移下，$d_L \approx d$。
3. 红移获取：宿主星系的光谱谱线移动给出红移 $z$。当 $z \ll 1$ 时，退行速度 $v \approx c z$。

本项目采用的 Tonry 2003 数据集收录了低红移 ($z < 0.1$) 的 Ia 型超新星样本。这些数据处于纯哈勃流区域，受暗能量影响极小，因此是校准 $H_0$ 绝对数值的理想样本。通过对这批数据构建稳健的统计模型（加权最小二乘、最大似然与贝叶斯 MCMC），我们可以提取出高置信度的 $H_0$ 测量值，为理解宇宙的膨胀史提供可靠的距离阶梯锚点。
## 贡献

欢迎提交 Issue 和 Pull Request 来改进拟合方法或增加新的数据集。

## 致谢

- 数据来源于 Tonry et al. (2003) 的Ia超新星项目。
- 感谢 `emcee` 团队提供的卓越采样工具。
- 感谢 github.com/wj198414/ASTRON1221 项目
