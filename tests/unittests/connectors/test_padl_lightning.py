import sys
import pytest
import torch

import padl
from padl import transform
try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from padl import PADLLightning
except ImportError:
    pass


@transform
class PadlEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128), torch.nn.ReLU(), torch.nn.Linear(128, 3)
        )

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding


@transform
class PadlDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(3, 128), torch.nn.ReLU(), torch.nn.Linear(128, 28 * 28)
        )

    def forward(self, x):
        decoding = self.decoder(x)
        return decoding


# TODO Saving in test_padl_lightning fails because padl_loss cannot be found when declared here.
@transform
def padl_loss(reconstruction, original):
    return torch.nn.functional.mse_loss(reconstruction, original)


@pytest.mark.skipif('pytorch_lightning' not in sys.modules,
                    reason="requires the Pytorch Lightning library")
def test_padl_lightning(tmp_path):
    @transform
    def padl_loss(reconstruction, original):
        return torch.nn.functional.mse_loss(reconstruction, original)

    autoencoder = PadlEncoder() >> PadlDecoder()
    padl_training_model = (
        transform(lambda x: x.view(x.size(0), -1))
        >> autoencoder + padl.identity
        >> padl_loss
    )
    train_data = [torch.randn([28, 28])] * 16
    val_data = [torch.randn([28, 28])] * 16
    checkpoint_callback = ModelCheckpoint(dirpath=str(tmp_path))
    trainer = pl.Trainer(max_steps=10, default_root_dir=str(tmp_path), log_every_n_steps=2,
                         callbacks=[checkpoint_callback])
    # trainer = pl.Trainer(max_steps=10, default_root_dir=str(tmp_path), log_every_n_steps=2)
    padl_lightning = PADLLightning(padl_training_model, train_data=train_data, val_data=val_data,
                                   batch_size=2, num_workers=0)
    trainer.fit(padl_lightning)
