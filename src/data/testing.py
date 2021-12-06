import os
from pandas_plink import read_plink
import dask
os.chdir("data/raw")
import time

(bim, fam, G) = read_plink('mhcabg')
start = time.time()
G_stacked = dask.array.vstack([G]*2)
G_stacked2 = dask.array.hstack([G_stacked]*2).rechunk()
G_stacked2.T.compute()
e = time.time() - start

start = time.time()
(bim, fam, G) = read_plink('mhcabg')
G.astype("int8") # notably faster with INT!
G_stacked = dask.array.vstack([G]*2)
G_stacked2 = dask.array.hstack([G_stacked]*2).rechunk()
G_stacked2.compute()
e2 = time.time() - start
# G_stacked2.T.to_zarr("testing_transpose/arr.zarr")
# N = dask.array.from_zarr("testing_transpose/arr.zarr")





bim.shape # 26433 snps
fam.shape # 4942 samples

G[3000, :3000].compute()


path = os.path.join("..", ".." ,  "dsmwpred", "data", "ukbb", "geno")
path = os.path.join("..", ".." , "..", "..",  "dsmwpred", "data", "ukbb", "geno")
(bim, fam, G) = read_plink(path)

G.shape

bim.shape # 628694 snps
fam.shape # 392214 samples

G.T[:10].compute()

res = G.T[:1000].compute()


import time
import os
from pandas_plink import read_plink1_bin, Chunk
path = os.path.join("..", ".." ,   "dsmwpred", "data", "ukbb", "geno.bed")
G = read_plink1_bin(path, verbose=False, chunk=Chunk(nsamples=10, nvariants=None))
s = time.time()
out = G[:10].compute(scheduler='processes')
time_taken = time.time() - s


# BEST!!!
import os
from pandas_plink import read_plink
path = os.path.join("..", ".." ,   "dsmwpred", "data", "ukbb", "geno")
(bim, fam, G) = read_plink(path)
G[:, 5].compute() # 5 samples all SNP
G[:, 10 000].compute() # 10k samples all SNP WORKS!
