import pandas as pd
import torch

PHOIBLE_FEATURE_COLUMNS = [
    "tone",
    "stress",
    "syllabic",
    "short",
    "long",
    "consonantal",
    "sonorant",
    "continuant",
    "delayedRelease",
    "approximant",
    "tap",
    "trill",
    "nasal",
    "lateral",
    "labial",
    "round",
    "labiodental",
    "coronal",
    "anterior",
    "distributed",
    "strident",
    "dorsal",
    "high",
    "low",
    "front",
    "back",
    "tense",
    "retractedTongueRoot",
    "advancedTongueRoot",
    "periodicGlottalSource",
    "epilaryngealSource",
    "spreadGlottis",
    "constrictedGlottis",
    "fortis",
    "raisedLarynxEjective",
    "loweredLarynxImplosive",
    "click",
]

def _normalize_filter_values(values):
    if values is None:
        return None
    if isinstance(values, (list, tuple, set)):
        normalized = [str(value).strip() for value in values if str(value).strip()]
    else:
        normalized = [part.strip() for part in str(values).split(",") if part.strip()]
    return normalized or None

def phoible_value_to_float(value):
    if value == "+":
        return 1.0
    if value == "-":
        return -1.0
    return 0.0

def load_phoible_feature_dict(
    phoible_csv_path,
    glottocode=None,
    iso6393=None,
    inventory_id=None,
):
    """
    Membaca phoible.csv dan menghasilkan dict:
        {
            "a": tensor([...]),
            "ŋ": tensor([...]),
            ...
        }
    """
    df = pd.read_csv(phoible_csv_path)

    glottocodes = _normalize_filter_values(glottocode)
    iso6393_codes = _normalize_filter_values(iso6393)
    inventory_ids = _normalize_filter_values(inventory_id)

    if glottocodes is not None and "Glottocode" in df.columns:
        df = df[df["Glottocode"].astype(str).isin(glottocodes)]
    if iso6393_codes is not None and "ISO6393" in df.columns:
        df = df[df["ISO6393"].astype(str).isin(iso6393_codes)]
    if inventory_ids is not None and "InventoryID" in df.columns:
        df = df[df["InventoryID"].astype(str).isin(inventory_ids)]

    feature_dict = {}
    for _, row in df.iterrows():
        ipa = row.get("Phoneme")
        if not isinstance(ipa, str) or not ipa:
            continue
        feats = [phoible_value_to_float(row.get(column, 0.0)) for column in PHOIBLE_FEATURE_COLUMNS]
        feature_dict[ipa] = torch.tensor(feats, dtype=torch.float32)
    return feature_dict

def build_lpep_phoible_table(
    token_to_id,
    phoible_feature_dict,
):
    """
    Membuat tensor:
        [n_token, phon_feat_dim]

    Untuk setiap token:
        ipa = token
        jika ipa ada di phoible_feature_dict:
            table[idx] = phoible_feature_dict[ipa]
        else:
            table[idx] = zero vector
            tambahkan ke list missing
    Return:
        table, missing
    """
    if not token_to_id:
        raise ValueError("token_to_id must not be empty")

    feat_dim = len(PHOIBLE_FEATURE_COLUMNS)
    table = torch.zeros(len(token_to_id), feat_dim, dtype=torch.float32)
    missing = []

    for token, idx in token_to_id.items():
        if token in phoible_feature_dict:
            table[idx] = phoible_feature_dict[token]
        else:
            missing.append(token)

    return table, missing
