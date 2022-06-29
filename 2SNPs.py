#!/usr/bin/env python

import numpy as np
from scipy.stats import norm

n_snp = 2
n_rep = 1000
n_gwas = 10000
b_true = np.array([0.1, 0.01])
h2_set = 0.1
mvn_cov = np.array([[1, -0.9], [-0.9, 1]])
maf = np.array([0.5, 0.5])

gt_haplotype = np.random.multivariate_normal(np.zeros(n_snp), mvn_cov, (2, n_gwas, n_rep))
percent_point = norm.ppf(maf)
foo = gt_haplotype < percent_point
gt_haplotype[foo] = 1
gt_haplotype[~foo] = 0
gt_ind = gt_haplotype[0] + gt_haplotype[1]

genetic_effect = gt_ind @ b_true
genetic_effect_var = genetic_effect.var(ddof=1)
environment_effect_var = genetic_effect_var / h2_set * (1 - h2_set)
environment_effect = np.random.normal(0, environment_effect_var ** .5, (n_gwas, n_rep))
phenotype = genetic_effect + environment_effect
h2_observed = genetic_effect.var(axis=0, ddof=1) / phenotype.var(axis=0, ddof=1)

x = np.copy(gt_ind)
y = np.copy(phenotype)
x -= x.mean(axis=0)
y -= y.mean(axis=0)
xx = np.sum(x ** 2, axis=0)
yy = np.sum(y ** 2, axis=0)
xy = np.sum(x * np.atleast_3d(y), axis=0)

b = xy / xx
sigma2 = (yy.reshape(n_rep, 1) - b * xy) / (n_gwas - 2)
b_se = np.sqrt(sigma2 / xx)
z = b / b_se

sd_x = gt_ind.std(ddof=1, axis=0)
sd_z2 = np.sqrt(4 * z ** 2 - 2 + np.array(0j))
ld = np.empty((n_rep, n_snp, n_snp))
ld2 = np.empty((n_rep, n_snp, n_snp))
ld_inv = np.empty((n_rep, n_snp, n_snp))
ld2_inv = np.empty((n_rep, n_snp, n_snp))
sd_x_outer = np.empty((n_rep, n_snp, n_snp))
sd_z2_outer = np.empty((n_rep, n_snp, n_snp))
b_gwas_sign_outer = np.empty((n_rep, n_snp, n_snp))
for a in range(n_rep):
    ld[a] = np.corrcoef(gt_ind[:, a, :], rowvar=False)
    ld2[a] = ld[a] ** 2
    ld_inv[a] = np.linalg.pinv(ld[a], hermitian=True)
    ld2_inv[a] = np.linalg.pinv(ld2[a], hermitian=True)
    sd_x_outer[a] = np.real(np.outer(sd_x[a], sd_x[a]))
    sd_z2_outer[a] = np.real(np.outer(sd_z2[a], sd_z2[a]))
    b_gwas_sign = np.sign(b[a])
    b_gwas_sign_outer[a] = np.outer(b_gwas_sign, b_gwas_sign)

print('ld correlation coefficient', np.percentile(ld[:, 0, 1], [0, 50, 100]))

h2_true = (b_true @ (ld * sd_x_outer) @ b_true) / phenotype.var(ddof=1, axis=0)
h2_true_deviation = (h2_true - h2_observed) / h2_observed * 100
print('h2_true_deviation', np.percentile(h2_true_deviation, [0, 50, 100]))

h2_hess = (z.reshape(n_rep, 1, n_snp) @ ld_inv @ z.reshape(n_rep, n_snp, 1)).reshape(-1) / n_gwas
h2_hess /= 1 + h2_hess
h2_hess_se = np.sqrt((2 * n_snp * ((1 - h2_hess) / n_gwas) + 4 * h2_hess) * ((1 - h2_hess) / n_gwas))
h2_hess_deviation = (h2_hess - h2_observed) / h2_observed * 100
print('h2_hess_deviation', np.percentile(h2_hess_deviation, [0, 50, 100]))

b_true_sign = np.sign(b_true)
b_true_sign_outer = np.outer(b_true_sign, b_true_sign)
h2_ehe = np.sum(ld2_inv @ np.atleast_3d((z ** 2 - 1) / n_gwas), axis=1)
h2_ehe /= 1 + h2_ehe
z = np.atleast_3d(z)
z_var = 4 * z @ np.swapaxes(z, 1, 2) * ld - 2 * ld2
h2_ehe_sd = np.sqrt(np.sum(ld2_inv @ z_var @ ld2_inv, axis=(1, 2)) / n_gwas ** 2)
h2_ehe_deviation = (h2_ehe - h2_observed) / h2_observed * 100
print('h2_ehe_deviation', np.percentile(h2_ehe_deviation, [0, 50, 100]))

print(h2_ehe.std(), h2_ehe_sd.mean())
