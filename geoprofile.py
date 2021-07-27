import sys
import libgeoprofile as gp
from os.path import join
import torch
import numpy as np
from torch.utils.data import IterableDataset
from numpy.random import default_rng


__all__ = ['Geoprofile']

def board_path(base_dir, board_id, face):
    return join(base_dir, "{}.GeoMap.{}.ltiBuffer".format(board_id, face))


def bcc_path(base_dir, board_id):
    return join(base_dir, "{}.bcc".format(board_id))


def dataset_load_paths(base_dir, board_id, bcc=False):
    return [
        board_path(base_dir, board_id, "Top"),
        board_path(base_dir, board_id, "Right"),
        board_path(base_dir, board_id, "Bottom"),
        board_path(base_dir, board_id, "Left"),
        bcc_path(base_dir, board_id) if bcc else None,
    ]

class _GeoprofileDataset(IterableDataset):
    def __init__(self, base_dir, points_per_row, rows_per_chunk, list_file, shuffle=True, augment_cfg_path=None):
        self.base_dir = base_dir
        self.points_per_row = points_per_row
        self.rows_per_chunk = rows_per_chunk
        self.list_file = list_file
        self.shuffle = shuffle
        if augment_cfg_path:
            try:
                self.augment_config = gp.load_augment_config(augment_cfg_path)
            except:
                print(f"Failed to load augment config at {augment_cfg_path}, writing defaults and quitting.")
                gp.write_default_augment_config(augment_cfg_path)
                raise
        else:
            self.augment_config = None

    def __iter__(self):
        rng = default_rng()
        with open(join(self.base_dir, self.list_file)) as train_list_file:
            for line in train_list_file:
                board_id = line[:-1]
                try:
                    args = dataset_load_paths(self.base_dir, board_id, True)
                    args.append(self.points_per_row)
                    args.append(self.rows_per_chunk)
                    if self.augment_config:
                        board = gp.augment_from_paths(*args, self.augment_config)
                    else:
                        board = gp.load_from_paths(*args)
                    points, labels = board.batches_labelled(1) # Batch size of 1 so that we can shuffle the chunks of this board
                except Exception as err:
                    print(f"Warning: unable to load board {board_id} from {self.base_dir}; {err}")
                    continue

                labels = labels.astype(np.int64).squeeze()
                points = points.squeeze()
                if self.shuffle:
                    indices = rng.permutation(len(points))
                    for idx in indices:
                        yield points[idx], labels[idx]
                else:
                    for p, l in zip(points, labels):
                        yield p, l 


class Geoprofile(dict):
    def __init__(self, base_dir, points_per_row, rows_per_chunk, shuffle=True, augment_cfg_path=None):
        """ Create a new dataset, located at `base_dir`, and reading from `train.txt` and `test.txt` to determine which board IDs to pick. Shuffling is always disabled for the test dataset. """
        super().__init__()
        self['train'] = _GeoprofileDataset(base_dir, points_per_row, rows_per_chunk, "train.txt", shuffle, augment_cfg_path)
        self['test'] = _GeoprofileDataset(base_dir, points_per_row, rows_per_chunk, "test.txt", False, None)
