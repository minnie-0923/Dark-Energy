# Data preprocessing for the supernova cosmology analysis.


# ===========================================
# Hubble constant measurement: cosmological data analysis
# Fit Hubble's law, v = H0 * d, with the Tonry 2003 dataset.
# ===========================================

import numpy as np
import matplotlib.pyplot as plt

# Astronomy data-processing tools
import astropy.io.ascii  # reads astronomy tables
import astropy.units as u  # physical units
import astropy.constants as ac  # physical constants

# Upload the data file in Google Colab.
from google.colab import files
uploaded = files.upload()
# ===========================================
# 1. Data loading and preprocessing
# ===========================================

# Read the VOTable-style astronomy dataset.
dat = astropy.io.ascii.read("Tonry_2003.vot")

# print(dat)

# Compute distance data.
# col8 is a logarithmic distance quantity; 72.0 is the initial H0
# assumption in km/s/Mpc. 10**dat["col8"] converts it to a linear scale.
distance = 10**dat["col8"] / 72.0 * u.mpc

# Compute distance uncertainty from col9.
distance_error = (10**(dat["col8"] + dat["col9"]) - 10**dat["col8"]) / 72.0 * u.mpc

# Compute velocity data; col7 is stored in logarithmic form.
velocity = 10**dat["col7"] * u.km / u.s
