"""Full DOA-only Mamba-CNN model."""

from torch import nn

from .mamba_signal_adapter import MambaSignalAdapter
from .doa_covariance_cnn import DOACovarianceCNNHead


class DOAMambaNet(nn.Module):
    """Compose I/Q signal adaptation and covariance/CNN DOA classification.

    Receives (B, IQ, M, T) and returns (B, num_doa_classes) raw logits.
    The caller supplies dataset and model dictionaries read from configuration.
    """

    def __init__(self, dataset_config, model_config):
        super().__init__()
        self.signal_adapter = MambaSignalAdapter(
            dataset_config=dataset_config,
            model_config=model_config,
        )
        self.doa_head = DOACovarianceCNNHead(
            dataset_config=dataset_config,
            model_config=model_config,
        )

    def forward(self, x):
        s_adapted = self.signal_adapter(x)
        doa_logits = self.doa_head(s_adapted)
        return doa_logits
