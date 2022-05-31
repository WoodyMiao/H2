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
