"""Connector to Pytorch Lightning"""

import torch

import pytorch_lightning as pl


class PADLLightning(pl.LightningModule):
    def __init__(
        self,
        padl_model,
        train_data,
        val_data,
        test_data,
        loader_kwargs
    ):
        super().__init__()
        self.model = padl_model
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.loader_kwargs = loader_kwargs

        self.model.pd_forward_device_check()

        # Set Pytorch layers as attributes from PADL model
        layers = padl_model.pd_layers
        for i, layer in enumerate(layers):
            key = f'layer_{i}'
            setattr(self, key, layer)

    def train_dataloader(self):
        return self.model._pd_get_loader(self.train_data, self.model.pd_preprocess, 'train',
                                         **self.loader_kwargs)

    def val_dataloader(self):
        return self.model._pd_get_loader(self.val_data, self.model.pd_preprocess, 'eval',
                                         **self.loader_kwargs)

    def test_dataloader(self):
        return self.model._pd_get_loader(self.test_data, self.model.pd_preprocess, 'eval',
                                         **self.loader_kwargs)

    def forward(self, x):
        """In lightning, forward defines the prediction/inference actions"""
        return None

    def training_step(self, batch, batch_idx):
        loss = self.model.pd_forward.pd_call_transform(batch, 'train')
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model.pd_forward.pd_call_transform(batch, 'eval')
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
