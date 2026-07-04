# Dark Energy: Measuring the Hubble Constant with Type Ia Supernovae

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

This project analyzes the Tonry et al. (2003) Type Ia supernova dataset to estimate the Hubble constant, \(H_0\), using linear regression, maximum likelihood estimation, and Bayesian MCMC sampling. The goal is to show how observational astronomy, statistical modeling, and cosmology connect: raw redshift-distance data become evidence about the expansion rate of the universe.

## Project Overview

The project uses low-redshift Type Ia supernova observations to test Hubble's law and estimate \(H_0\).

- Dataset: redshift and distance-modulus measurements from Tonry et al. (2003).
- Physical model: Hubble's law, \(v = H_0 d\).
- Statistical goal: estimate the best-fit value of \(H_0\) and quantify its uncertainty.
- Research focus: compare ordinary linear fitting, weighted least squares, maximum likelihood estimation, and Bayesian posterior sampling.

## Repository Structure

```bash
Dark-Energy/
├── data/
│   └── Tonry_2003.vot          # Tonry et al. observational table
├── src/
│   ├── data.py                 # Data loading and preprocessing
│   ├── visualization.py        # Initial visual exploration
│   ├── linear.py               # Linear and weighted least-squares fitting
│   ├── MLE.py                  # Maximum likelihood estimation
│   └── MCMC.py                 # Bayesian MCMC sampling
├── notebooks/
│   └── darkenergy.ipynb        # Main analysis notebook
├── results/                    # Exported figures
├── requirements.txt
└── README.md
```

## Quick Start

1. Clone the repository.

   ```bash
   git clone https://github.com/minnie-0923/Dark-Energy.git
   cd Dark-Energy
   ```

2. Install dependencies.

   ```bash
   pip install -r requirements.txt
   ```

3. Open the main notebook.

   ```bash
   jupyter notebook notebooks/darkenergy.ipynb
   ```

## Analysis Workflow

### 1. Data Preprocessing

- Read the VOTable-style astronomy dataset with `astropy.io.ascii`.
- Convert the distance-modulus-like column into linear distance in Mpc.
- Convert redshift-related velocity information into km/s.
- Propagate distance errors from logarithmic space into linear distance uncertainty.

### 2. Visual Exploration

- Plot velocity versus distance on logarithmic axes to inspect the full dynamic range.
- Plot the same relation on linear axes with error bars to see observational uncertainty.
- Select a 0-700 Mpc low-redshift range for fitting so that Hubble's law is a reasonable approximation.

### 3. Linear and Weighted Least-Squares Fitting

The first model fits the relation \(v = H_0 d\) directly. Because distance uncertainty is larger than velocity uncertainty, the later weighted fit uses distance as the dependent variable:

\[
d = m v + b,\quad H_0 = \frac{1}{m}.
\]

Weighted least squares uses the inverse of the distance variance as a precision weight:

\[
\mathbf{w} = (A^T C^{-1} A)^{-1} A^T C^{-1} \mathbf{y}.
\]

This gives an initial estimate of \(H_0\) and an analytic uncertainty from the covariance matrix.

### 4. Maximum Likelihood Estimation

The MLE model introduces an intrinsic-scatter factor, \(f\), to account for additional dispersion not explained by measurement error alone. The total variance is modeled as

\[
\sigma^2 = \sigma_{\rm meas}^2 + (\text{model}\cdot f)^2.
\]

The Gaussian log-likelihood is

\[
\log \mathcal{L} =
-\frac{1}{2}\sum_i
\left[
\frac{(y_i-\text{model}_i)^2}{\sigma_i^2}
+ \log(\sigma_i^2)
\right].
\]

`scipy.optimize.minimize` is used to find the maximum-likelihood parameters, which then become the starting point for the MCMC walkers.

### 5. Bayesian MCMC Sampling

The project uses Bayes' theorem:

\[
P(\theta \mid \text{data}) \propto P(\text{data}\mid\theta)P(\theta).
\]

The posterior is evaluated in log space for numerical stability:

\[
\log P(\theta \mid \text{data}) =
\log P(\text{data}\mid\theta) + \log P(\theta).
\]

The MCMC implementation uses `emcee`:

- 32 walkers
- 5,000 iterations
- burn-in removal
- thinning every 15 samples
- autocorrelation-time checks
- posterior visualization with a corner plot

## Representative Results

The notebook reports these example values from the included run:

- Weighted least squares: \(H_0 \approx 67.80 \pm 0.51\) km/s/Mpc.
- Maximum likelihood: \(H_0 \approx 71.61\) km/s/Mpc.
- MCMC: posterior samples for \(m\), \(b\), and \(\log f\), with a corner plot showing the correlation between slope and intercept.

The difference between the WLS and MLE/MCMC estimates is not a failure; it shows how assumptions about uncertainty and intrinsic scatter change the inferred cosmological parameter.

## Physical Background

Modern cosmology is built on general relativity and the cosmological principle. At low redshift, the expansion of the universe can be approximated by Hubble's law:

\[
v = H_0 d.
\]

Here \(v\) is recession velocity, \(d\) is distance, and \(H_0\) is the present-day expansion rate of the universe. Measuring \(H_0\) is central to cosmology because it connects local distance-ladder observations with early-universe measurements from the cosmic microwave background.

Type Ia supernovae are especially useful because they can be standardized as "standard candles." Their observed brightness gives a distance estimate, and their host-galaxy spectra give redshift. The Tonry et al. (2003) sample therefore provides a compact way to practice how cosmological parameters are inferred from astronomical observations.

## Skills Demonstrated

- Astrophysical data preprocessing with `astropy`
- Dimensional reasoning with physical units
- Error propagation from logarithmic measurements
- Linear regression and weighted least squares
- Maximum likelihood estimation
- Bayesian inference with MCMC
- Posterior diagnostics and visualization

## Acknowledgments

- Data source: Tonry et al. (2003) Type Ia supernova observations.
- Sampling library: `emcee`.
- Posterior visualization: `corner`.
- Reference inspiration: github.com/wj198414/ASTRON1221.
