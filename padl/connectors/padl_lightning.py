"""Connector to Pytorch Lightning"""

import torch

import pytorch_lightning as pl


class PADLLightning(pl.LightningModule):
    """Connector to Pytorch Lightning

    :param padl_model:
    :param train_data: list of training data points
    :param val_data: list of validation data points
    :param test_data: list of test data points
    :param loader_kwargs: loader key word arguments for the DataLoader
    """
    def __init__(
        self,
        padl_model,
        train_data,
        val_data=None,
        test_data=None,
        **kwargs
    ):
        super().__init__()
        self.model = padl_model
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.loader_kwargs = kwargs

        self.model.pd_forward_device_check()

        # Set Pytorch layers as attributes from PADL model
        layers = padl_model.pd_layers
        for i, layer in enumerate(layers):
            key = f'layer_{i}'
            setattr(self, key, layer)

    def forward(self, x):
        """In lightning, forward defines the prediction/inference actions"""
        return None

    def train_dataloader(self):
        """Create the train dataloader using `pd_get_loader`"""
        return self.model.pd_get_loader(self.train_data, self.model.pd_preprocess, 'train',
                                        **self.loader_kwargs)

    def val_dataloader(self):
        """Create the val dataloader using `pd_get_loader` if *self.val_data* is provided"""
        if self.val_data is not None:
            return self.model.pd_get_loader(self.val_data, self.model.pd_preprocess, 'eval',
                                            **self.loader_kwargs)
        return None

    def test_dataloader(self):
        """Create the test dataloader using `pd_get_loader` if *self.test_data* is provided"""
        if self.test_data is not None:
            return self.model.pd_get_loader(self.test_data, self.model.pd_preprocess, 'eval',
                                            **self.loader_kwargs)
        return None

    def training_step(self, batch, batch_idx):
        """Default training step"""
        loss = self.model.pd_forward.pd_call_transform(batch, 'train')
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Default validation step"""
        loss = self.model.pd_forward.pd_call_transform(batch, 'eval')
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        """Default test step"""
        loss = self.model.pd_forward.pd_call_transform(batch, 'eval')
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
