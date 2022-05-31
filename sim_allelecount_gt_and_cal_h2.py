#!/usr/bin/env python

import time
import logging
import argparse
import numpy as np
import pandas as pd
from scipy.stats import norm
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr


class RefPanel(object):
    def __init__(self, partition_bed, plink_pre):
        self.bed = np.loadtxt(partition_bed, dtype=int, skiprows=1)
        self.map = np.loadtxt(f'{plink_pre}.map', dtype=int, usecols=(0, 3))

        with open(f'{plink_pre}.ped') as f:
            ncols = len(f.readline().split(' '))

        plink_ped = np.loadtxt(f'{plink_pre}.ped', dtype=int, usecols=range(6, ncols))
        self.haplotype = np.concatenate((plink_ped[:, ::2], plink_ped[:, 1::2]))

    def get_locus(self, partition_idx):
        in_block_idx = np.where(
            (self.map[:, 0] == self.bed[partition_idx, 0]) &
            (self.map[:, 1] >= self.bed[partition_idx, 1]) &
            (self.map[:, 1] <= self.bed[partition_idx, 2])
        )[0]

        haplotype = self.haplotype[:, in_block_idx]
        haplotype = haplotype[:, np.all((haplotype == 0) | (haplotype == 1), axis=0)]
        nsnp_block = haplotype.shape[1]

        if nsnp_block < n_snp:
            raise Exception(f'Only {nsnp_block} valid SNPs in block {partition_idx}')

        logging.info(f'{nsnp_block} valid SNPs in block {partition_idx}, use {n_snp} SNPs as ref panel.')
        start = (nsnp_block - n_snp) / 2
        return haplotype[:, int(start):int(nsnp_block - start)]


class Simulation(object):
    def __init__(self, gt_ref, h2_true, n_ld_panel):
        self.ld = None
        self.h2_true = h2_true

        self.gt_gwas = self.simulate_allelecount_gt(gt_ref, n_gwas)
        if n_ld_panel is None:
            self.gt_ld_panel = self.gt_gwas[:, 0]
        else:
            self.gt_ld_panel = self.simulate_allelecount_gt(gt_ref, n_ld_panel)[:, 0]

    @staticmethod
    def simulate_allelecount_gt(gt_ref, n_ind):
        importr('hapsim')
        haplodata = robjects.r('haplodata')
        m = robjects.r.matrix(robjects.IntVector(gt_ref.T.reshape(-1)), nrow=gt_ref.shape[0])
        x = haplodata(m)
        x = dict(zip(x.names, list(x)))
        mvn_cov = np.array(x['cor'])
        gt_sim = np.random.multivariate_normal(np.zeros(mvn_cov.shape[0]), mvn_cov, (2, n_ind, n_rep))
        frq_ref = gt_ref.mean(axis=0)
        percent_point = norm.ppf(frq_ref)
        foo = gt_sim < percent_point
        gt_sim[foo] = 1
        gt_sim[~foo] = 0
        return gt_sim[0] + gt_sim[1]

    def ld_eigendecomposition(self, max_num_eig, min_eigval):
        ld = np.corrcoef(self.gt_ld_panel, rowvar=False)
        ld_rank = np.linalg.matrix_rank(ld)
        if ld_rank == n_gwas:
            raise Exception(f'Rank of the LD matrix equals the sample size.')

        ld_eigval, ld_eigvec = np.linalg.eigh(ld)
        idx = ld_eigval.argsort()[::-1]
        ld_eigval = ld_eigval[idx]
        ld_eigvec = ld_eigvec[:, idx]
        k = min(max_num_eig, (ld_eigval > min_eigval).sum())
        ld2inv = np.linalg.pinv(ld ** 2, hermitian=True)

        self.ld = ld
        return ld_eigval, ld_eigvec, ld_rank, k, ld2inv

    def beta_3snps_effect_phenotype(self, beta_u, beta_w):
        u, v, w = 0, int(n_snp / 2), -1
        u_var, v_var, w_var = self.gt_ld_panel[:, [u, v, w]].var(axis=0, ddof=1)

        a = v_var
        b = 2 * beta_u * self.ld[u, v] * np.sqrt(u_var * v_var) + 2 * beta_w * self.ld[w, v] * np.sqrt(w_var * v_var)
        c = u_var * beta_u ** 2 + w_var * beta_w ** 2 + \
            2 * beta_u * beta_w * self.ld[u, w] * np.sqrt(u_var * w_var) - self.h2_true
        delta = b ** 2 - 4 * a * c

        if delta < 0:
            raise Exception('Delta < 0')

        beta_v = (- b + delta ** 0.5) / (2 * a)
        beta = np.zeros(self.gt_ld_panel.shape[1])
        beta[[u, v, w]] = beta_u, beta_v, beta_w

        genetic_effect = self.gt_gwas @ beta
        environment_effect = np.random.normal(0, np.sqrt(1 - self.h2_true), (n_gwas, n_rep))
        phenotype = genetic_effect + environment_effect
        h2_observed = genetic_effect.var(axis=0, ddof=1) / phenotype.var(axis=0, ddof=1)

        return beta, phenotype, h2_observed


class GWASandH2(object):
    def __init__(self, geno, pheno):
        self.x = geno
        self.y = pheno
        self.z = None

    def gwas(self):
        x = np.copy(self.x)
        y = np.copy(self.y)
        x -= x.mean(axis=0)
        y -= y.mean(axis=0)
        xx = np.sum(x ** 2, axis=0)
        yy = np.sum(y ** 2, axis=0)
        xy = np.sum(x * np.atleast_3d(y), axis=0)

        b = xy / xx
        sigma2 = (yy.reshape(n_rep, 1) - b * xy) / (n_gwas - 2)
        b_se = np.sqrt(sigma2 / xx)
        self.z = b / b_se

    def h2(self, ld_eigval_, ld_eigvec_, ld_rank_, k_, ld2inv_):
        g_beta_k = (((self.z / n_gwas ** 0.5) @ ld_eigvec_[:, :k_]) ** 2 / ld_eigval_[:k_]).sum(axis=1)
        h2_hess = ((n_gwas * g_beta_k - k_) / (n_gwas - k_)).reshape(n_rep, 1)
        vg = (self.z ** 2 - 1) / (n_gwas + self.z ** 2 - 1)
        h2_kgg = np.sum(ld2inv_ @ vg.reshape(n_rep, n_snp, 1), axis=1)

        heritability = (h2_hess, self.h2_se(h2_hess, ld_rank_), h2_kgg, self.h2_se(h2_kgg, ld_rank_))
        cols = ['h2_hess', 'h2_hess_se', 'h2_kggsee', 'h2_kggsee_se']
        df_heritability = pd.DataFrame(np.concatenate(heritability, axis=1), columns=cols)
        return df_heritability

    @staticmethod
    def h2_se(h2, p):
        var = (n_gwas / (n_gwas - p)) ** 2 * (2 * p * ((1 - h2) / n_gwas) + 4 * h2) * ((1 - h2) / n_gwas)
        var[var < 0] = np.nan
        return var ** .5


def analyze_one_partititon(partition_idx):
    logging.info(f'Simulate genotypes of partition {partition_idx} '
                 f'at {time.strftime("%d %b %Y %H:%M:%S", time.localtime())}')
    refpanel_gt = refpanel.get_locus(partition_idx)
    np.savetxt(f'{args.out_pre}_{partition_idx}.refpanel.tsv', refpanel_gt, delimiter='\t', fmt='%d', comments='')

    simulation = Simulation(refpanel_gt, args.h2_true, args.n_ld_panel)
    ld_eigval, ld_eigvec, ld_rank, k, ld2inv = simulation.ld_eigendecomposition(args.max_num_eig, args.min_eigval)
    np.savetxt(f'{args.out_pre}_{partition_idx}.ld.tsv', simulation.ld, delimiter='\t', fmt='%.4g', comments='')

    beta, phenotype, h2_observed = simulation.beta_3snps_effect_phenotype(beta_u=0.1, beta_w=0.2)
    np.savetxt(f'{args.out_pre}_{partition_idx}.true_beta.tsv', beta.reshape(1, -1),
               delimiter='\t', fmt='%.4g', comments='')

    logging.info(f'Perform association tests for partition {partition_idx} '
                 f'at {time.strftime("%d %b %Y %H:%M:%S", time.localtime())}')
    gwas_h2 = GWASandH2(simulation.gt_gwas, phenotype)
    gwas_h2.gwas()

    logging.info(f'Calculate heritability of partition {partition_idx} '
                 f'at {time.strftime("%d %b %Y %H:%M:%S", time.localtime())}')
    results = gwas_h2.h2(ld_eigval, ld_eigvec, ld_rank, k, ld2inv)
    results['h2_observed'] = h2_observed
    logging.info(f'Output results of partition {partition_idx} '
                 f'at {time.strftime("%d %b %Y %H:%M:%S", time.localtime())}')
    results.to_csv(f'{args.out_pre}_{partition_idx}.h2.tsv', sep='\t', index=False, float_format='%.4g')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate genotypes by HapSim and estimate h2 by HESS and KGGSEE')
    parser.add_argument('--partition-bed', type=str, required=True, help='Genome partition in BED format')
    parser.add_argument('--partition-idx', type=int, default=None, help='Index of the target partition in the BED file'
                                                                        'If not specified, apply to all partitions.')
    parser.add_argument('--plink-pre', type=str, required=True, help='Prefix of PLINK text file for reference panel')
    parser.add_argument('--out-pre', type=str, required=True, help='Output file prefix')
    parser.add_argument('--h2-true', type=float, required=True, help='True heritability to be simulated')
    parser.add_argument('--n-snp', type=int, default=10, help='Number of SNPs to be simulated')
    parser.add_argument('--n-gwas', type=int, default=1000, help='Sample size of GWAS to be simulated')
    parser.add_argument('--n-ld-panel', type=int, default=None, help='Sample size of LD panel to be simulated.'
                                                                     'If not specified, the first GWAS sample '
                                                                     'will also be used as the LD panel.')
    parser.add_argument('--n-rep', type=int, default=100, help='Number of repetitions to be simulated.'
                                                               'This is for empirical SE calculation.')
    parser.add_argument('--maf-threshold', type=float, default=0.05, help='Minimum minor allele frequency')
    parser.add_argument('--min-eigval', type=float, default=1, help='Minimum eigenvalue')
    parser.add_argument('--max-num-eig', type=int, default=50, help='Maximum number of eigenvalues')

    args = parser.parse_args()
    n_snp = args.n_snp
    n_rep = args.n_rep
    n_gwas = args.n_gwas

    logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
    logging.info(f'Start reading reference panel at {time.strftime("%d %b %Y %H:%M:%S", time.localtime())}')
    refpanel = RefPanel(args.partition_bed, args.plink_pre)

    if args.partition_idx is None:
        for i in range(refpanel.bed.shape[0]):
            try:
                analyze_one_partititon(i)
            except Exception as e:
                print(e)
    else:
        analyze_one_partititon(args.partition_idx)
