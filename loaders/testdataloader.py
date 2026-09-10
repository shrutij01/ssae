"""
Simple, safe data loading with validation.
"""

import os
import torch
import numpy as np
from typing import Dict, Tuple, Optional, Union, List
import yaml
from box import Box


class TestDataLoader:
    """Load data safely with validation."""

    def __init__(
        self, device: Union[str, torch.device] = "cpu", verbose: bool = False
    ):
        self.device = (
            torch.device(device) if isinstance(device, str) else device
        )
        self.verbose = verbose

    def load_test_data(self, datafile: str, dataconfig_path: str) -> Tuple[
        Optional[Tuple[torch.Tensor, torch.Tensor]],
        Optional[Box],
        Optional[List],
        str,
    ]:
        """
        Load test data with validation.

        Returns: (tensors, config, labels, status_message)
        """
        try:
            # Load config
            config = self._load_config(dataconfig_path)
            if config is None:
                return None, None, None, "Config loading failed"

            # Load raw data
            raw_data = self._load_raw_data(datafile)
            if raw_data is None:
                return None, config, None, "Raw data loading failed"

            tilde_z_raw, z_raw = raw_data

            # Convert to tensors on target device
            import utils.data_utils as utils

            tilde_z_tensor = utils.tensorify(tilde_z_raw, self.device)
            z_tensor = utils.tensorify(z_raw, self.device)

            # Check dimensions
            if len(z_tensor.shape) != 2 or z_tensor.shape[1] != config.rep_dim:
                return None, config, None, "Invalid dimensions"

            # Load labels
            labels = self._load_labels(datafile)
            if labels is None:
                return None, config, None, "Label loading failed"

            # Validate compatibility
            if len(z_tensor) != len(labels):
                print(
                    f"Data/label length mismatch {len(z_tensor)} vs {len(labels)}"
                )
                labels = labels[-len(z_tensor) :]
                # to handle when labels are for entire dataset
                # fixme: better handling

            return (tilde_z_tensor, z_tensor), config, labels, "Success"

        except Exception as e:
            return None, None, None, str(e)

    def _load_config(self, path: str) -> Optional[Box]:
        """Load YAML config file."""
        try:
            if not os.path.exists(path):
                return None
            with open(path, "r") as f:
                config_dict = yaml.safe_load(f)
            return Box(config_dict) if config_dict else None
        except Exception:
            return None

    def _load_raw_data(self, datafile: str) -> Optional[Tuple]:
        """Load raw data using utils."""
        try:
            import utils.data_utils as utils

            if not os.path.exists(datafile):
                return None
            return utils.load_test_data(datafile=datafile)
        except Exception:
            return None

    def _load_labels(self, datafile: str) -> Optional[List]:
        """Load concept labels."""
        try:
            import utils.data_utils as utils

            labels_dir = os.path.dirname(os.path.abspath(datafile))
            labels_file = os.path.join(labels_dir, "concept_labels_test.json")

            if not os.path.exists(labels_file):
                return None

            labels = utils.load_json(labels_file)
            return labels if labels else None
        except Exception:
            return None

    def split_by_label(
        self, tilde_z: torch.Tensor, z: torch.Tensor, labels: List
    ) -> Optional[Dict]:
        """Split data by concept labels."""
        try:
            if len(z) != len(labels):
                return None

            labels_array = np.array(labels)
            unique_labels = np.unique(labels_array)
            split_data = {}

            for label in unique_labels:
                indices = np.where(labels_array == label)[0]
                if len(indices) == 0:
                    continue

                indices_tensor = torch.tensor(
                    indices, device=z.device, dtype=torch.long
                )

                # Bounds check
                if indices_tensor.max().item() >= len(z):
                    return None

                z_subset = z[indices_tensor]
                tilde_z_subset = tilde_z[indices_tensor]

                split_data[label] = (tilde_z_subset, z_subset)

            return split_data if split_data else None

        except Exception:
            return None
