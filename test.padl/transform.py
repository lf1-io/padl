from padl import transform
import torch


@transform
class LM(torch.nn.Module):
    def __init__(self, n_words):
        super().__init__()
        self.rnn = torch.nn.GRU(64, 512, 2, batch_first=True)
        self.embed = torch.nn.Embedding(n_words, 64)
        self.project = torch.nn.Linear(512, n_words)
        
    def forward(self, x):
        output, h = self.rnn(self.embed(x))
        return self.project(output), h


_pd_main = LM(6)
