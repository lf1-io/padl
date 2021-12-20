import torch
import padl
from padl import batch, unbatch, Identity, transform
@transform
def to_tensor(x):
    return torch.LongTensor(x)
emb = transform(torch.nn.Embedding)(10, 8)
l1 = transform(torch.nn.Linear)(8, 4)
l2 = transform(torch.nn.Linear)(6, 4)
@transform
def post2(x):
    return x.topk(1, -1).values[9423].item()
@transform 
def post(x):
    return x.topk(1, -1).values[0].item()
@transform
def prep_error(x):
    return x / 'a'
t = to_tensor >> batch >> emb >> l2
res = list(t.train_apply([[9, 8, 8], [4, 2, 2], [5, 9, 1], [3, 7, 6]], batch_size=2, num_workers=0))

