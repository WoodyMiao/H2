#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
from pysnptools.snpreader import Bed

nsnp = 50
nind = 1000
nrep = 1000
h2_true = 0.1
min_eigval = 1
max_num_eig = 1000
refpanel_plink_pre = 'test/EUR_chr22'
partition_bed_file = 'test/block_chr22.bed'
partition_index = 10


def load_refpanel_ld_block():
    partition = np.loadtxt(partition_bed_file, dtype=int, skiprows=1)
    bfile = Bed(refpanel_plink_pre, count_A1=False)
    bfile_genotype = bfile.read().val

    in_block_idx = np.where(
        (bfile.pos[:, 0] == partition[partition_index, 0]) &
        (bfile.pos[:, 2] >= partition[partition_index, 1]) &
        (bfile.pos[:, 2] <= partition[partition_index, 2])
    )[0]

    refpanel_genotype = bfile_genotype[:, in_block_idx]
    refpanel_genotype = refpanel_genotype[:, ~np.isnan(refpanel_genotype).any(axis=0)]
    maf = refpanel_genotype.mean(axis=0) / 2
    maf[maf > 0.5] = 1.0 - maf[maf > 0.5]
    refpanel_genotype = refpanel_genotype[:, maf > 0.05]

    nsnp_block = refpanel_genotype.shape[1]
    if nsnp_block < nsnp:
        print(f'There are only {nsnp_block} SNPs in the LD block, while {nsnp} was asked.')
        sys.exit(1)

    start = (nsnp_block - nsnp) / 2
    ld_ = np.corrcoef(refpanel_genotype[:, int(start):int(nsnp_block - start)], rowvar=False)
    q_ = np.linalg.matrix_rank(ld_)
    if q_ == nind:
        print(f'Rank of the LD matrix equals the sample size.')
        sys.exit(1)

    ld_eigval_, ld_eigvec_ = np.linalg.eigh(ld_)
    idx = ld_eigval_.argsort()[::-1]
    ld_eigval_ = ld_eigval_[idx]
    ld_eigvec_ = ld_eigvec_[:, idx]
    k_ = min(max_num_eig, (ld_eigval_ > min_eigval).sum())
    ld2inv_ = np.linalg.pinv(ld_ ** 2, hermitian=True)

    return ld_, ld_eigval_, ld_eigvec_, q_, k_, ld2inv_


def beta_with_three_causal_snps(beta_u=0.1, beta_v=0.1, u=0, v=nsnp - 1, x=int(nsnp / 2)):
    # Suppose SNP i, j and x are causal.
    b = 2 * beta_u * ld[u, x] + 2 * beta_v * ld[v, x]
    c = beta_u ** 2 + beta_v ** 2 + 2 * beta_u * beta_v * ld[u, v] - h2_true
    delta = b ** 2 - 4 * c

    if delta < 0:
        print('Delta < 0')
        sys.exit(1)

    beta_x = (- b + delta ** 0.5) / 2
    beta = np.zeros(nsnp)
    beta[[u, v, x]] = beta_u, beta_v, beta_x

    return beta


def estimate_gwas_z(x, y):
    x -= x.mean(axis=0)
    y -= y.mean(axis=0)
    xx = np.sum(x ** 2, axis=0)
    yy = np.sum(y ** 2, axis=0)
    xy = np.sum(x.reshape(nind, nrep, nsnp) * y.reshape(nind, nrep, 1), axis=0)

    b = xy / xx
    sigma2 = (yy.reshape(nrep, 1) - b * xy) / (nind - 2)
    b_se = np.sqrt(sigma2 / xx)
    return b / b_se


def h2_se(h2, p):
    var = (nind / (nind - p)) ** 2 * (2 * p * ((1 - h2) / nind) + 4 * h2) * ((1 - h2) / nind)
    var[var < 0] = np.nan
    return var ** .5


ld, ld_eigval, ld_eigvec, q, k, ld2inv = load_refpanel_ld_block()
beta_fixed = beta_with_three_causal_snps()

genotypes = np.random.multivariate_normal(np.zeros(nsnp), ld, (nind, nrep))
genetic_effect = genotypes @ beta_fixed
environment_effect = np.random.normal(0, np.sqrt(1 - h2_true), (nind, nrep))
phenotype = genetic_effect + environment_effect
h2_observed = genetic_effect.var(axis=0, ddof=1) / phenotype.var(axis=0, ddof=1)

z = estimate_gwas_z(genotypes, phenotype)
beta_gwas = z / nind ** 0.5
g_beta_k = ((beta_gwas @ ld_eigvec[:, :k]) ** 2 / ld_eigval[:k]).sum(axis=1)
h2_hess = ((nind * g_beta_k - k) / (nind - k)).reshape(nrep, 1)
vg = (z ** 2 - 1) / (nind + z ** 2 - 1)
h2_kgg = np.sum(ld2inv @ vg.reshape(nrep, nsnp, 1), axis=1)

results = (h2_hess, h2_se(h2_hess, q), h2_se(h2_hess, k), h2_kgg, h2_se(h2_kgg, q), h2_se(h2_kgg, k))
cols = ['h2_hess', 'h2_hess_se_q', 'h2_hess_se_k', 'h2_kggsee', 'h2_kggsee_se_q', 'h2_kggsee_se_k']
df = pd.DataFrame(np.concatenate(results, axis=1), columns=cols)
