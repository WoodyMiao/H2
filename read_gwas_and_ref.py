import numpy as np
import pandas as pd
from pysnptools.snpreader import Bed


class GWASandRefPanel(object):
    """
    Handles reference panel and GWAS summary statistics
    """

    def __init__(self, bfile_prefix, sumstats_file, min_maf):
        # read reference genotype, impute, and get maf
        bim = pd.read_csv(f'{bfile_prefix}.bim', sep=r'\s+', header=None,
                          names=['CHR', 'snp', 'cm', 'BP', 'a1', 'a2'], dtype={'CHR': str})
        gt = Bed(bfile_prefix, sid=bim.snp, count_A1=False).read().val

        bim['raw_idx'] = np.arange(bim.shape[0])
        bim = bim[bim.CHR.str.isnumeric()].copy()
        bim.CHR = bim.CHR.astype(int)
        bim.set_index(['CHR', 'BP'], inplace=True)
        print(f'Read {bim.shape[0]} SNPs from autosomes of reference panel.')

        nanidx = np.where(np.isnan(gt))
        mean_geno = np.nanmean(gt, axis=0)
        gt[nanidx] = mean_geno[nanidx[1]]
        maf = gt.sum(axis=0) / gt.shape[0] / 2
        maf[maf > 0.5] = 1.0 - maf[maf > 0.5]

        # read gwas
        gwas = pd.read_csv(sumstats_file, sep=r'\s+', dtype={'CHR': str})
        gwas = gwas[gwas.CHR.str.isnumeric()].copy()
        gwas.CHR = gwas.CHR.astype(int)
        gwas.set_index(['CHR', 'BP'], inplace=True)
        print(f'Read {gwas.shape[0]} SNPs from autosomes of GWAS summary statistics.')

        # merge and filter
        gwas = gwas[(~gwas.index.duplicated(keep=False)) & gwas.index.isin(bim.index)]
        bim = bim[(maf >= min_maf) & (~bim.index.duplicated(keep=False))]
        gwas = pd.concat([gwas, bim], axis=1, join='inner').sort_index()
        gwas = gwas[((gwas.A1 == gwas.a1) & (gwas.A2 == gwas.a2)) |
                    ((gwas.A1 == gwas.a2) & (gwas.A2 == gwas.a1))].reset_index()

        self.ref = gt[:, gwas['raw_idx']]
        self.gwas = gwas[['CHR', 'BP', 'Z']]
        print(f'{gwas.shape[0]} SNPs left after merging and filtering.')

    def get_locus(self, chrom, start, stop):
        # Returns the legend and genotype matrix at a locus, specified by start (inclusive) and stop (exclusive)
        gwas = self.gwas.loc[(self.gwas.CHR == chrom) & (self.gwas.BP >= start) & (self.gwas.BP < stop)]
        return gwas.Z.values, self.ref[:, gwas.index], gwas.shape[0]


def compute_h2_one_locus(z, nind, ld=None, min_eigval=1, max_num_eig=1000):
    q = np.linalg.matrix_rank(ld)
    if q == nind:
        return

    # compute the ld matrix and its eigen decomposition
    ld_eigval, ld_eigvec = np.linalg.eigh(ld)
    idx = ld_eigval.argsort()[::-1]
    ld_eigval = ld_eigval[idx]
    ld_eigvec = ld_eigvec[:, idx]

    # compute hess h2 estimate
    beta = z / nind ** .5
    k = min(max_num_eig, (ld_eigval > min_eigval).sum())
    quad_form = ((beta @ ld_eigvec)[:k] ** 2 / ld_eigval[:k]).sum()
    h2_adj = (nind * quad_form - k) / (nind - k)

    # compute kggsee h2 estimate
    ld2 = ld ** 2
    ld2_inv = np.linalg.pinv(ld2, hermitian=True)
    vg = (z ** 2 - 1) / (nind + z ** 2 - 1)
    h2_kgg = np.sum(ld2_inv @ vg)

    # compute se and p
    def se(h2, p):
        var = (nind / (nind - p)) ** 2 * (2 * p * ((1 - h2) / nind) + 4 * h2) * ((1 - h2) / nind)
        return var ** .5

    return q, k, h2_adj, se(h2_adj, q), se(h2_adj, k), h2_kgg, se(h2_kgg, q), se(h2_kgg, k)


def compute_h2_multipart(partition, gwas_refpanel, nind):
    # iterate through loci
    results = list()
    n_partition = partition.shape[0]
    partition.columns = ['chrom', 'start', 'stop']

    for i in range(n_partition):
        chrom, start, stop = partition.iloc[i]
        # extract data in the locus
        z, refgt, nsnp = gwas_refpanel.get_locus(chrom, start, stop)
        print(f'Calculating h2 of {nsnp} SNPs in chr{chrom}:{start}-{stop} (locus {i + 1} / {n_partition}) ...')
        if nsnp == 0:
            continue
        ld = np.corrcoef(refgt, rowvar=False)
        results.append((chrom, start, stop, nsnp) + compute_h2_one_locus(z, nind, ld))

    header = ['chrom', 'start', 'stop', 'num_snp', 'rank', 'k', 'h2_unb', 'h2_unb_se', 'h2_unb_p',
              'h2_adj', 'h2_adj_se', 'h2_adj_p', 'h2_kgg', 'h2_kgg_se', 'h2_kgg_p']

    return pd.DataFrame(results, columns=header)


partition_fnm = 'test/block_chr22.bed'
refpanel_fnm = 'test/EUR_chr22'
sumstats_fnm = 'test/SCZ_chr22.tsv'
out_fnm = 'test/SCZ_chr22_h2.tsv'
nsnp = 99863

partition = pd.read_csv(partition_fnm, sep=r'\s+')
gwas_refpanel = GWASandRefPanel(refpanel_fnm, sumstats_fnm, min_maf=0.05)
df = compute_h2_multipart(partition, gwas_refpanel, nsnp)
df.to_csv(out_fnm, sep='\t', float_format='%.3g', index=False)