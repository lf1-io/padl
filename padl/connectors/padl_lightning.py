"""Connector to Pytorch Lightning"""

import torch

import pytorch_lightning as pl


class PADLLightning(pl.LightningModule):
    def __init__(
        self,
        padl_model,
        train_data,
        val_data,
        test_data
    ):
        super().__init__()
        self.model = padl_model
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        # Set Pytorch layers as attributes from PADL model
        layers = padl_model.pd_layers
        for i, layer in enumerate(layers):
            key = f'layer_{i}'
            setattr(self, key, layer)

    def train_dataloader(self):
        return self.model._pd_get_loader(self.train_data, self.model.pd_preprocess, 'train',
                                         batch_size=8, num_workers=4)

    def val_dataloader(self):
        return self.model._pd_get_loader(self.val_data, self.model.pd_preprocess, 'eval',
                                         batch_size=8, num_workers=4)

    def test_dataloader(self):
        return self.model._pd_get_loader(self.test_data, self.model.pd_preprocess, 'eval',
                                         batch_size=8, num_workers=4)

    def forward(self, x):
        """In lightning, forward defines the prediction/inference actions"""
        return None

    def training_step(self, batch, batch_idx):
        loss = self.model.pd_forward._pd_call_transform(batch, 'train')
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model.pd_forward._pd_call_transform(batch, 'eval')
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
