# Dark-Energy
learn something about astrophysics!

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)


一个用于分析 Tonry 2003 超新星数据的 Python 项目，通过线性回归、最大似然估计（MLE）和贝叶斯 MCMC 方法精确测量哈勃常数 $H_0$。

## 项目概述

本项目利用 Ia 型超新星的观测数据来验证哈勃定律，并估算哈勃常数 $H_0$：

- **数据集**: 使用 Tonry 2003 公布的红移-距离模数数据集。
- **物理模型**: 哈勃定律 $v = H_0 \times d$。
- **核心目标**: 获得 $H_0$ 的最优估计及其不确定性。





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

1. **克隆仓库**

   ```bash
   git clone https://github.com/minnie-0923/Minnie-Astrophysics.git
   cd Minnie-Astrophysics
   ```

2. **安装依赖**

   ```bash
   pip install -r requirement.txt
   ```

### 基本使用

运行主分析笔记本：

```python
# 打开Jupyter笔记本
jupyter notebook notebooks/GW150914_Analysis.ipynb
```

## 分析流程

### 1. 数据获取

从GWOSC获取汉福德(H1)和利文斯顿(L1)观测站的应变数据

### 2. 频谱分析

- 计算振幅谱密度(ASD)
- 识别60Hz、120Hz、180Hz电源线干扰
- 分析噪声特性

### 3. 信号滤波

- 50-250Hz带通滤波器保留引力波特征频段
- 陷波滤波器消除电源线干扰
- 零相位滤波避免信号失真

### 4. 时域分析

- 滤波前后信号对比
- 应变振幅随时间变化
- 事件附近信号细节

### 5. 时频分析

- Q变换显示频率随时间变化
- 啁啾信号特征可视化
- 能量分布分析

### 6. 多站关联

- 汉福德与利文斯顿数据对比
- 时间延迟补偿(6.9ms)
- 相位调整和信号相关性分析

### 7. 音频生成

- 信号归一化处理
- WAV文件格式转换
- 可听化引力波信号

## 技术细节

### 主要依赖库

- **GWPy**: 专业引力波数据分析
- **NumPy/SciPy**: 科学计算和信号处理
- **Matplotlib**: 数据可视化

### 滤波器设计

```python
# 带通滤波器：50-250Hz
bp = filter_design.bandpass(50, 250, hdata.sample_rate)

# 陷波滤波器：消除电源线干扰
notches = [filter_design.notch(line, hdata.sample_rate) for line in (60, 120, 180)]
```

## 结果示例

项目生成多种分析结果：

- 振幅谱密度图
- 滤波前后信号对比
- Q变换时频图
- 多探测器数据关联
- 引力波音频文件

## 物理背景

基于广义相对论，引力波是时空弯曲的涟漪：

- 双黑洞并合产生强烈引力波
- 信号特征：旋近、合并、铃荡三个阶段
- 频率啁啾和振幅增长特性

## 贡献

欢迎提交Issue和Pull Request来改进项目

## 致谢

- LIGO科学合作组织提供开放数据
- GWPy开发团队提供专业分析工具
- 所有为引力波探测做出贡献的科学家
- github.com/wj198414/ASTRON1221的项目启发
