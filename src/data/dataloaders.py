from typing import Iterable
import torch
from torch.utils.data import IterableDataset, DataLoader

import dask.array as da
import random


class DaskMapDataset(IterableDataset):
    def __init__(self, array: da.Array) -> IterableDataset:
        self.array = array

    def __len__(self):
        return self.array.shape[0]

    def __getitem__(self, idx: int):
        return self.array[idx].compute()


class DaskIterableDataset(IterableDataset):
    def __init__(self, array: da.Array, buffer_size: int = 1024) -> IterableDataset:
        self.array = array
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterable:
        n, _ = self.array.shape

        starts = range(0, n, self.buffer_size)
        ends = range(self.buffer_size - 1, n, self.buffer_size)

        for start, end in zip(starts, ends):
            X = torch.Tensor(self.array[start, end].compute())
            for x in X:
                yield x
        if n > end:
            X = torch.Tensor(self.array[end, n].compute())
            for x in X:
                yield x


class ShuffleDataset(IterableDataset):
    def __init__(self, dataset, buffer_size: int = 1024):
        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        shufbuf = []
        try:
            dataset_iter = iter(self.dataset)
            for i in range(self.buffer_size):
                shufbuf.append(next(dataset_iter))
        except:
            self.buffer_size = len(shufbuf)

        try:
            while True:
                try:
                    item = next(dataset_iter)
                    evict_idx = random.randint(0, self.buffer_size - 1)
                    yield shufbuf[evict_idx]
                    shufbuf[evict_idx] = item
                except StopIteration:
                    break
            while len(shufbuf) > 0:
                yield shufbuf.pop()
        except GeneratorExit:
            pass


if __name__ == "__main__":
    import os
    import time

    path = os.path.join("data", "interim", "chrom_1")
    arr = da.from_zarr(path)  # shape (participants, snps)

    dataset = DaskIterableDataset(array=arr)
    dataset = ShuffleDataset(dataset, 1024)
    dl = DataLoader(dataset, batch_size=64)

    s = time.time()
    [d for d in iter(dl)]
    e = time.time() - s
    print("shuffle took {e}")

    dataset = DaskMapDataset(array=arr)
    dl = DataLoader(dataset, batch_size=64)
    s = time.time()
    [d for d in iter(dl)]
    e = time.time() - s
    print("map took {e}")
