import numpy as np
import pandas as pd
from darts.models import RNNModel
from darts.utils.data import DualCovariatesShiftedDataset
import torch
from darts import TimeSeries
from typing import Optional

class PaddedDualCovariatesShiftedDataset(DualCovariatesShiftedDataset):
    """Wrapper around Darts' dataset with padding support"""
    
    def __getitem__(self, idx: int):
        # Get original item
        item = super().__getitem__(idx)
        
        # Pad sequences to fixed length
        input_target = np.pad(item[0], (0, self.input_chunk_length - len(item[0])), mode='constant')
        future_covariates = np.pad(item[2], (0, self.input_chunk_length - len(item[2])), mode='constant')
        
        # Convert to tensors
        return (
            torch.tensor(input_target, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(item[1], dtype=torch.float32).unsqueeze(-1),  # historic_covariates
            torch.tensor(future_covariates, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(item[3], dtype=torch.float32).unsqueeze(-1)   # target
        )

class EventRNNModel(RNNModel):
    """
    Custom RNN model with padding support that passes dataset checks.
    """
    def __init__(self, model='LSTM', dropout=0.0, hidden_dim=256, *args, **kwargs):
        print("Initializing EventRNNModel ...")
        kwargs['model'] = model
        kwargs['dropout'] = dropout
        kwargs['hidden_dim'] = hidden_dim
        super().__init__(*args, **kwargs)


    def _verify_train_dataset_type(self, train_dataset):
        """Override dataset type check to allow our padded dataset."""
        return isinstance(train_dataset, PaddedDualCovariatesShiftedDataset)

    def _setup_training_data(
        self, 
        target_sequence: TimeSeries,
        past_covariates: Optional[TimeSeries],
        future_covariates: Optional[TimeSeries],
        max_samples_per_ts: Optional[int],
    ) -> DualCovariatesShiftedDataset:
        """Custom padded dataset"""
        return PaddedDualCovariatesShiftedDataset(
            target_series=target_sequence,
            covariates=past_covariates,
            future_covariates=future_covariates,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            max_samples_per_ts=max_samples_per_ts
        )

    def _patched_produce_train_output(self, input_batch: tuple):
        """Process padded batches and ignore required dummies."""
        past_target, historic_covariates, future_covariates, static_covariates = input_batch
        output = past_target.unsqueeze(-1) if past_target.ndim == 2 else past_target
        output.requires_grad_(True)
        return output

    def forward(self, x: torch.Tensor, future_covariates: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Custom forward pass through the raw LSTM and projection layer.
        Preserves gradient history.
        """
        rnn_out, _ = self.model.rnn(x)     # shape: (batch, seq_len, hidden_dim)
        output = self.model.V(rnn_out)     # shape: (batch, seq_len, 1)
        return output                      # still has requires_grad = True



    def training_step(self, train_batch, batch_idx):
        input_target, _, future_covariates, target = train_batch
        # Call the forward() method defined above. This keeps the gradient history.
        output = self(input_target, future_covariates)
        loss = self.criterion(output, target)
        self.log("train_loss", loss, prog_bar=True)
        return loss

