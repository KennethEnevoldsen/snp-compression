import torch

from torch.utils.data import IterableDataset, DataLoader


data = list(range(20))

class MyIterableDataset(IterableDataset):
    def __init__(self, data):
        self.data = data
    
    def __iter__(self):
        for i in self.data:
            yield i

dataset = MyIterableDataset(data)

loader = DataLoader(dataset, batch_size=3)

for batch in loader:
    print(batch)

import datasets

datasets.Features(
    {
        "id": datasets.Value("string"),
        "title": datasets.Value("string"),
        "context": datasets.Value("string"),
        "question": datasets.Value("string"),
        "answers": datasets.Sequence(
            {
                "text": datasets.Value("string"),
                "answer_start": datasets.Value("int32"),
            }
        ),
    }
)