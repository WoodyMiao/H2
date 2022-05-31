#!/usr/bin/env python

import pandas as pd
from read_gwas_and_ref import GWASandRefPanel
from compute_h2 import compute_h2_multipart


partition_fnm = 'test/block_chr22.bed'
refpanel_fnm = 'test/EUR_chr22'
sumstats_fnm = 'test/SCZ_chr22.tsv'
out_fnm = 'test/SCZ_chr22_h2.tsv'
nsnp = 99863

partition = pd.read_csv(partition_fnm, sep=r'\s+')
gwas_refpanel = GWASandRefPanel(refpanel_fnm, sumstats_fnm, min_maf=0.05)
df = compute_h2_multipart(partition, gwas_refpanel, nsnp)
df.to_csv(out_fnm, sep='\t', float_format='%.3g', index=False)
