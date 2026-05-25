from monotonic_align import maximum_path
from monotonic_align import mask_from_lens
from monotonic_align.core import maximum_path_c
import numpy as np
import torch
import copy
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import matplotlib.pyplot as plt
from munch import Munch

def maximum_path(neg_cent, mask):
  """ Cython optimized version.
  neg_cent: [b, t_t, t_s]
  mask: [b, t_t, t_s]
  """
  device = neg_cent.device
  dtype = neg_cent.dtype
  neg_cent =  np.ascontiguousarray(neg_cent.data.cpu().numpy().astype(np.float32))
  path =  np.ascontiguousarray(np.zeros(neg_cent.shape, dtype=np.int32))

  t_t_max = np.ascontiguousarray(mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32))
  t_s_max = np.ascontiguousarray(mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32))
  maximum_path_c(path, neg_cent, t_t_max, t_s_max)
  return torch.from_numpy(path).to(device=device, dtype=dtype)

def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = "Data/train_list.txt"
    if val_path is None:
        val_path = "Data/val_list.txt"

    with open(train_path, 'r', encoding='utf-8', errors='ignore') as f:
        train_list = f.readlines()
    with open(val_path, 'r', encoding='utf-8', errors='ignore') as f:
        val_list = f.readlines()

    return train_list, val_list

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

# for norm consistency loss
def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x

def get_image(arrs):
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(arrs)

    return fig

def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d
    
def log_print(message, logger):
    logger.info(message)
    print(message)

def load_lpep_feature_table(args):
    use_lpep = getattr(args, "use_lpep", False)
    use_phoible_features = getattr(args, "use_phoible_features", False)
    if not (use_lpep and use_phoible_features):
        return None

    table_path = getattr(args, "phoible_feature_table_path", None)
    if not table_path:
        raise ValueError("use_phoible_features=True but phoible_feature_table_path is not provided")

    obj = torch.load(table_path, map_location="cpu")
    if isinstance(obj, dict):
        feature_table = obj.get("feature_table")
    else:
        feature_table = obj
    if feature_table is None:
        raise KeyError(f"feature_table not found in {table_path}")
    feature_table = feature_table.float()

    expected_tokens = getattr(args, "n_token", None)
    if expected_tokens is not None and feature_table.size(0) != expected_tokens:
        raise ValueError(
            f"PHOIBLE feature table token count mismatch: expected {expected_tokens}, got {feature_table.size(0)}"
        )

    expected_feat_dim = getattr(args, "phon_feat_dim", None)
    if expected_feat_dim is not None and feature_table.size(1) != expected_feat_dim:
        raise ValueError(
            f"PHOIBLE feature table feature dim mismatch: expected {expected_feat_dim}, got {feature_table.size(1)}"
        )

    return feature_table

def build_lpep_inputs(tokens, args, phoible_feature_table=None, lang_id=None):
    batch_size = tokens.size(0)
    device = tokens.device

    if lang_id is None:
        lang_id = torch.zeros(batch_size, dtype=torch.long, device=device)
    elif not torch.is_tensor(lang_id):
        lang_id = torch.tensor(lang_id, dtype=torch.long, device=device)
    else:
        lang_id = lang_id.to(device=device, dtype=torch.long)
    if lang_id.dim() == 0:
        lang_id = lang_id.expand(batch_size)
    lang_id = lang_id.view(batch_size)

    phon_feats = None
    if getattr(args, "use_lpep", False) and getattr(args, "use_phoible_features", False):
        if phoible_feature_table is None:
            raise ValueError("use_phoible_features=True but PHOIBLE feature table is not loaded")
        phon_feats = phoible_feature_table.to(device=device)[tokens]

    return lang_id, phon_feats
    
