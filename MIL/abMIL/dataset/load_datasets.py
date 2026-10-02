from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import torch


class PatchDataset(Dataset):
    """Patch-level bags: one .pt per slide holding a [N_patches, D] feature tensor.
    CSV paths are relative to ``feature_root``.

    Returns (coords, features, label, file_name, ood_label). coords is an empty
    placeholder (patch coordinates are not used by the model); it keeps the tuple
    layout shared with the slide-level loader. label is -1 and ood_label is 1 for
    OOD slides; ood_label is 0 when the CSV has no such column."""

    def __init__(self, csv_file, feature_root, datatype):
        self.csv_file = pd.read_csv(csv_file)
        self.feature_root = feature_root
        self.datatype = datatype
        self.has_ood = 'ood_label' in self.csv_file.columns

        if self.datatype == 'train':
            self.csv_index = [self.csv_file.columns.get_loc('train'),self.csv_file.columns.get_loc('train_label')]
            self.csv_file = self.csv_file[self.csv_file['train'].notna()].reset_index(drop=True)
            self.lenth = self.csv_file['train'].count()
        elif self.datatype == 'val':
            self.csv_index = [self.csv_file.columns.get_loc('val'),self.csv_file.columns.get_loc('val_label')]
            self.csv_file = self.csv_file[self.csv_file['val'].notna()].reset_index(drop=True)
            self.lenth = self.csv_file['val'].count()
        elif self.datatype == 'test':
            self.csv_index = [self.csv_file.columns.get_loc('test'),self.csv_file.columns.get_loc('test_label')]
            self.csv_file = self.csv_file[self.csv_file['test'].notna()].reset_index(drop=True)
            self.lenth = self.csv_file['test'].count()

        if self.has_ood:
            self.ood_index = self.csv_file.columns.get_loc('ood_label')

    def __len__(self):
        return self.lenth
    
    def __getitem__(self, index):
        file_name = self.csv_file.iloc[index, self.csv_index[0]]
        label = self.csv_file.iloc[index, self.csv_index[1]]
        features = torch.load(os.path.join(self.feature_root, file_name), map_location='cpu')
        coords = torch.empty(0)

        ood_label = int(self.csv_file.iloc[index, self.ood_index]) if self.has_ood else 0
        return coords, features, label, file_name, ood_label


class SlideDataset(Dataset):
    """Dataset for slide-level embeddings (one .pt per slide, e.g. CHIEF_WSI).

    Each .pt file contains a single slide-level embedding tensor of shape [D] or [1, D].
    The CSV path column stores paths relative to ``feature_root``.

    Returns:
        embedding  : Tensor [D]      — slide-level feature vector
        label      : int             — class label (-1 for OOD slides)
        file_path  : str             — path to the .pt file (relative to feature_root)
        ood_label  : int             — 0 = in-distribution, 1 = OOD
    """

    def __init__(self, csv_file: str, feature_root: str, datatype: str):
        """
        Args:
            csv_file     : path to the fold CSV produced by create_dataset/*.py
            feature_root : root directory the CSV paths are relative to
            datatype     : 'train' | 'val' | 'test'
        """
        df = pd.read_csv(csv_file)
        self.feature_root = feature_root

        col_path  = datatype               # e.g. 'train', 'val', 'test'
        col_label = f'{datatype}_label'    # e.g. 'train_label'

        df = df[df[col_path].notna()].reset_index(drop=True)

        self.paths     = df[col_path].values
        self.labels    = df[col_label].values
        self.ood_labels = (df['ood_label'].values if 'ood_label' in df.columns
                           else np.zeros(len(df), dtype=int))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        file_path = str(self.paths[index])
        label     = int(self.labels[index])
        ood_label = int(self.ood_labels[index])

        embedding = torch.load(os.path.join(self.feature_root, file_path), map_location='cpu')
        embedding = embedding.detach() if isinstance(embedding, torch.Tensor) else embedding

        # Normalise shape: [1, D] or [D] -> [D]
        if isinstance(embedding, torch.Tensor):
            if embedding.dim() == 2:
                embedding = embedding.squeeze(0)   # [1, D] -> [D]
        else:
            # dict or other format — attempt to extract the feature tensor
            raise ValueError(
                f"Unexpected embedding format {type(embedding)} from {file_path}. "
                "Expected a torch.Tensor of shape [D] or [1, D]."
            )

        return embedding, label, file_path, ood_label
