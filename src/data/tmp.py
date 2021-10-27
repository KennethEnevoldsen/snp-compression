import os
from typing import Tuple
from plinkio import plinkfile
import torch
from more_itertools import chunked
import time
from wasabi import msg

plink_path = os.path.join("..", "..", "dsmwpred", "data", "ukbb", "geno")
transpose_path = os.path.join("data", "interim")

plink = plinkfile.open(plink_path)

samples = plink.get_samples()

n_samples = len(samples)
n_loci = len(plink.get_loci())
batch = 1000


def batch_plink_to_tensor(interval: Tuple[int, int], n_loci: int=n_loci) -> torch.Tensor:
    plink = plinkfile.open(plink_path)
    s= time.time()
    stop = False
    batch_size = interval[1] - interval[0]
    X = torch.zeros(batch_size, n_loci, dtype=torch.int8)
    for r, row in enumerate(plink):
            # and write file
        for c, geno in enumerate(row):
            if c < (interval[0]-1):
                continue
            if c == interval[1]:
                stop = True
            X[r][c] = geno
        if stop is True:
            break
    plink.close()
    return X

def interval_gen(n_samples, step = 3000):
    for i in range(0, n_samples, step):
        if i == 0: 
            ii = 0
        else:
            yield (ii, i)
            ii = i +1

intervals = list(interval_gen(n_samples, step = 50_000))


ss = time.time()
for interval in intervals[:10]:
    s = time.time()
    X = batch_plink_to_tensor(interval, n_loci)
    e = time.time()-s
    print(interval)
    print("\t Time: ", e)

e = time.time()-ss
print("Total Time: ", e)

# s = time.time()
# batch = 1000
# geno = chunked(([genotype for genotype in row] for row in plink), batch)
# for i in range(int(1000/batch)):
#     g = next(geno)
#     X = torch.tensor(g)

# e2 = time.time()-s
# # 25.74967646598816




len(plink.get_samples())

def tester(chunk_size):
    geno = chunked(([genotype for genotype in row] for row in plink), chunk_size)
    pheno = chunked((s.phenotype for s in plink.get_samples()), chunk_size)

    i = 0
    total_time = 0
    for g, p in zip(geno, pheno):
        i += 1

        s = time.time()

        x = torch.tensor(g)
        y = torch.tensor(p)

        e = time.time() - s
        total_time += e

        if (i * chunk_size) > 1000 and (i % 10 == 0):
            msg.info(f"Time taken (chunk_size = {chunk_size}):")
            print(f"\tTotal time: {total_time}")
            print(f"\tAvg. time pr. chunk: {total_time/i}")
            print(f"\tAvg. time pr. sample: {total_time/(i*chunk_size)}")
            break


# for i in [50, 100, 500, 1000]:
#     tester(chunk_size=i)


# ℹ Time taken (chunk_size = 50):
#         Total time: 21.1005756855011
#         Avg. time pr. chunk: 0.7033525228500366
#         Avg. time pr. sample: 0.014067050457000732
# ℹ Time taken (chunk_size = 100):
#         Total time: 27.969476461410522
#         Avg. time pr. chunk: 1.398473823070526
#         Avg. time pr. sample: 0.01398473823070526
# ℹ Time taken (chunk_size = 500):
#         Total time: 71.56262803077698
#         Avg. time pr. chunk: 7.156262803077698
#         Avg. time pr. sample: 0.014312525606155396
# ℹ Time taken (chunk_size = 1000):
#         Total time: 142.8147988319397
#         Avg. time pr. chunk: 14.28147988319397
#         Avg. time pr. sample: 0.014281479883193969

# Multiprocessing

# limit = 3000
chunk_size = 100


def to_tensor(v):
    g, p = v
    x = torch.tensor(g)
    y = torch.tensor(p)


def tester2(limit, chunk_size=100):
    def row_plink():
        for i, row in enumerate(plink):
            if i > limit:
                break
            yield row

    def sample_plink():
        for i, s in enumerate(plink.get_samples()):
            if i > limit:
                break
            yield s

    geno = chunked(([genotype for genotype in row] for row in row_plink()), chunk_size)
    pheno = chunked((s.phenotype for s in sample_plink()), chunk_size)
    from multiprocessing import Pool, cpu_count

    n_cores = cpu_count()
    s = time.time()
    with Pool(n_cores) as p:
        out = p.map_async(to_tensor, iterable=zip(geno, pheno), chunksize=5)
    e = time.time() - s
    print(f"Time taken (limit: {limit}, chunksize: {chunk_size})")
    print(f"\t", e)


for i in [1000, 2000, 4000]:
    tester2(i, chunk_size=100)

# map_async
# Time taken (limit: 1000, chunksize: 100)
#          33.890382051467896
# Time taken (limit: 2000, chunksize: 100)
#          24.634198427200317
# Time taken (limit: 4000, chunksize: 100)
#          49.63838291168213

# imap_unordered (map chunksize = 1)
# Time taken (limit: 1000, (chunksize 100)
#          19.32836675643921
# Time taken (limit: 2000)
#          34.69457507133484
# Time taken (limit: 3000)
#          51.20875859260559