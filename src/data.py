# 这段代码是对数据进行预处理


# ===========================================
# 哈勃常数测量：宇宙学数据分析
# 使用Tonry 2003数据拟合哈勃定律 v = H₀ * d
# ===========================================

import numpy as np
import matplotlib.pyplot as plt

# 导入天文数据处理库
import astropy.io.ascii  # 用于读取天文数据表格
import astropy.units as u  # 物理单位处理
import astropy.constants as ac  # 物理常数

# 在Google Colab中上传数据文件
from google.colab import files
uploaded = files.upload()
# ===========================================
# 1. 数据加载和预处理
# ===========================================

# 读取VOTable格式的天文数据（包含星系距离和速度信息）
dat = astropy.io.ascii.read("Tonry_2003.vot")

# print(dat)

# 计算距离数据：
# col8是距离模数的对数形式，72.0是H₀的初始假设值（km/s/Mpc）
# 10**dat["col8"]将距离模数转换为线性尺度
distance = 10**dat["col8"] / 72.0 * u.mpc

# 计算距离误差（从col9中提取）
distance_error = (10**(dat["col8"] + dat["col9"]) - 10**dat["col8"]) / 72.0 * u.mpc

# 计算速度数据：col7是速度的对数形式
velocity = 10**dat["col7"] * u.km / u.s
