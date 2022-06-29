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
    """
    This class handles the reference panel.
    """
    def __init__(self, partition_bed, plink_pre):
        """
        This function loads the reference panel.

        :param partition_bed: the bed file defines partitions to be simulated
        :param plink_pre: the prefix of plink .ped and .map files of reference panel
        """
        self.bed = np.loadtxt(partition_bed, dtype=int, skiprows=1)
        self.map = np.loadtxt(f'{plink_pre}.map', dtype=int, usecols=(0, 3))

        with open(f'{plink_pre}.ped') as f:
            ncols = len(f.readline().split(' '))

        plink_ped = np.loadtxt(f'{plink_pre}.ped', dtype=int, usecols=range(6, ncols))
        self.haplotype = np.concatenate((plink_ped[:, ::2], plink_ped[:, 1::2]))

    def get_locus(self, partition_idx, n_snp):
        """
        :param partition_idx: the line index of the partition to be simulated
        :param n_snp: the number of SNPs to be simulated
        :return: the genotypes of one partition
        """

        in_block_idx = np.where(
            (self.map[:, 0] == self.bed[partition_idx, 0]) &
            (self.map[:, 1] >= self.bed[partition_idx, 1]) &
            (self.map[:, 1] <= self.bed[partition_idx, 2])
        )[0]

        haplotype = self.haplotype[:, in_block_idx]
        haplotype = haplotype[:, np.all((haplotype == 0) | (haplotype == 1), axis=0)]
        nsnp_block = haplotype.shape[1]

        print(flush=True)
        if n_snp is None:
            n_snp = nsnp_block
        elif nsnp_block < n_snp:
            raise Exception(f'Only {nsnp_block} valid SNPs in partition {partition_idx}. Skip the partition.')

        logging.info(f'{nsnp_block} valid SNPs in partition {partition_idx}, use {int(n_snp)} SNPs as ref panel.')
        start = (nsnp_block - n_snp) / 2
        return haplotype[:, int(start):int(nsnp_block - start)]


class Simulation(object):
    """
    This class handles the simulated dataset.
    """
    def __init__(self, gt_ref, h2_true, n_ld_panel):
        """
        This function loads the data and parameters for simulation.

        :param gt_ref: genotypes of reference panel
        :param h2_true: the true heritability to be simulated
        :param n_ld_panel: the sample size for ld panel
        """

        self.gt_gwas = self.simulate_allelecount_gt(gt_ref, n_gwas)
        if n_ld_panel is None:
            gt_ld_panel = self.gt_gwas[:, 0]
        else:
            gt_ld_panel = self.simulate_allelecount_gt(gt_ref, int(n_ld_panel))[:, 0]

        self.ld = np.corrcoef(gt_ld_panel, rowvar=False)
        self.gt_ld_panel = gt_ld_panel
        self.h2_true = h2_true

    @staticmethod
    def simulate_allelecount_gt(gt_ref, n_ind):
        """
        :param gt_ref: genotypes of reference panel
        :param n_ind: number of diploid individuals to be simulated
        :return: genotypes of simulated individuals
        """
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

    def simulate_beta_and_phenotype(self):
        """
        :return: true effect sizes, simulated phenotypes, observed heritability
        """
        n_snp = self.ld.shape[0]
        u, v, w = int(n_snp * 0.25), int(n_snp * 0.5), int(n_snp * 0.75)
        u_var, v_var, w_var = self.gt_ld_panel[:, [u, v, w]].var(axis=0, ddof=1)

        beta_u = self.h2_true
        beta_w = beta_u ** 0.5

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
    """
    This class performs association tests and calculates heritability.
    """
    def __init__(self, geno, pheno):
        """
        :param geno: simulated genotypes
        :param pheno: simulated phenotypes
        """
        self.x = geno
        self.y = pheno
        self.z = None

    def gwas(self):
        """
        calculate the z-scores of association tests
        """
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

    def h2(self, ld, min_eigval, max_num_eig):
        """
        :param ld: the ld panel
        :param min_eigval: the minimum eigen value for HESS
        :param max_num_eig: the maximum number of eigen values for HESS
        :return: heritability calculated by HESS and EHE
        """
        def hess_var(h2_local):
            h2_local_var = (n_gwas / (n_gwas - ld_rank)) ** 2 * \
                           (2 * ld_rank * ((1 - h2_local) / n_gwas) + 4 * h2_local) * \
                           ((1 - h2_local) / n_gwas)
            return h2_local_var.reshape(-1, 1)

        ld_rank = np.linalg.matrix_rank(ld)
        ld_eigval, ld_eigvec = np.linalg.eigh(ld)
        idx = ld_eigval.argsort()[::-1]
        ld_eigval = ld_eigval[idx]
        ld_eigvec = ld_eigvec[:, idx]
        k = min(max_num_eig, (ld_eigval > min_eigval).sum())
        ld2 = ld ** 2
        ld2inv = np.linalg.pinv(ld2, rcond=1e-6, hermitian=True)

        g_beta_k = (((self.z / n_gwas ** 0.5) @ ld_eigvec[:, :k]) ** 2 / ld_eigval[:k]).sum(axis=1)
        h2_hess = ((n_gwas * g_beta_k - k) / (n_gwas - k)).reshape(n_rep, 1)
        h2_hess_var = hess_var(h2_hess)

        h2_ehe = np.sum(ld2inv @ np.atleast_3d(self.z ** 2 - 1), axis=1) / n_gwas
        h2_ehe /= 1 + h2_ehe

        z = np.atleast_3d(self.z)
        z_var = 4 * z @ np.swapaxes(z, 1, 2) * ld - 2 * ld2
        h2_ehe_var = np.sum(ld2inv @ z_var @ ld2inv, axis=(1, 2)) / n_gwas ** 2

        return pd.DataFrame(
            np.concatenate((h2_hess, h2_hess_var, h2_ehe, h2_ehe_var.reshape(-1, 1)), axis=1),
            columns=['h2_hess', 'h2_hess_var', 'h2_ehe', 'h2_ehe_var'])


def analyze_one_partititon(partition_idx):
    """
    This function conducts simulation and association tests, and calculates heritability for one partition.
    :param partition_idx: the line index of the partition to be simulated
    :return: heritability calculated by HESS and EHE
    """
    refpanel_gt = refpanel.get_locus(partition_idx, args.n_snp)

    logging.info(f'Simulating genotypes of partition {partition_idx} ...')
    simulation = Simulation(refpanel_gt, args.h2_true, args.n_ld_panel)
    np.savetxt(f'{args.out_pre}_{partition_idx}.ld.tsv', simulation.ld, delimiter='\t', fmt='%.3g', comments='')

    beta, phenotype, h2_observed = simulation.simulate_beta_and_phenotype()
    np.savetxt(f'{args.out_pre}_{partition_idx}.true_beta.tsv', beta.reshape(1, -1),
               delimiter='\t', fmt='%.3g', comments='')

    logging.info(f'Performing association tests for partition {partition_idx} ...')
    gwas_h2 = GWASandH2(simulation.gt_gwas, phenotype)
    gwas_h2.gwas()

    logging.info(f'Calculating the heritability of partition {partition_idx} ...')
    results = gwas_h2.h2(simulation.ld, args.min_eigval, args.max_num_eig)
    results['h2_observed'] = h2_observed
    logging.info(f'Partition {partition_idx} has been done '
                 f'at {time.strftime("%d %b %Y %H:%M:%S", time.localtime())}.')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate genotypes by HapSim and estimate h2 by HESS and EHE')
    parser.add_argument('--partition-bed', type=str, required=True, help='Genome partition in BED format')
    parser.add_argument('--plink-pre', type=str, required=True, help='Prefix of PLINK text file for reference panel')
    parser.add_argument('--out-pre', type=str, required=True, help='Output file prefix')
    parser.add_argument('--partition-idx', type=str, default=None, help='Index of the target partition in the BED file'
                                                                        'If not specified, apply to all partitions.')
    parser.add_argument('--n-snp', type=float, default=None, help='Number of SNPs to be simulated. '
                                                                  'If not specified, use all SNPs in the partition.')
    parser.add_argument('--n-ld-panel', type=float, default=None, help='Sample size of LD panel to be simulated.'
                                                                       'If not specified, the first GWAS sample '
                                                                       'will also be used as the LD panel.')
    parser.add_argument('--h2-true', type=float, default=0, help='True heritability to be simulated')
    parser.add_argument('--maf-threshold', type=float, default=0.05, help='Minimum minor allele frequency')
    parser.add_argument('--min-eigval', type=float, default=1, help='Minimum eigenvalue')
    parser.add_argument('--max-num-eig', type=float, default=50, help='Maximum number of eigenvalues')
    parser.add_argument('--n-gwas', type=float, default=1000, help='Sample size of GWAS to be simulated')
    parser.add_argument('--n-rep', type=float, default=100, help='The number of simulation repetitions '
                                                                 'for empirical SE calculation.')

    args = parser.parse_args()
    n_rep = int(args.n_rep)
    n_gwas = int(args.n_gwas)

    logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
    logging.info(f'Getting started at {time.strftime("%d %b %Y %H:%M:%S", time.localtime())}')

    refpanel = RefPanel(args.partition_bed, args.plink_pre)
    if args.partition_idx is None:
        indices = list(range(refpanel.bed.shape[0]))
    else:
        indices = [int(a) for a in args.partition_idx.split(',')]
    for i in indices:
        try:
            df = analyze_one_partititon(i)
            df.to_csv(f'{args.out_pre}_{i}.h2.tsv', sep='\t', index=False, float_format='%.3g')
        except Exception as e:
            print(e, flush=True)

    logging.info(f'All partitions have been done at {time.strftime("%d %b %Y %H:%M:%S", time.localtime())}')
