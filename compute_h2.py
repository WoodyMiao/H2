import numpy as np
import pandas as pd


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
