import os
import click
import torch

from text_utils import dicts as token_to_id
from Modules.phonology.phoible import build_lpep_phoible_table, load_phoible_feature_dict

def _parse_csv_option(value):
    if value is None:
        return None
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    return parts or None

@click.command()
@click.option("--phoible_csv_path", default="data/phoible/phoible.csv", type=str)
@click.option("--output_path", default="data/phoible/lpep_phoible_table.pt", type=str)
@click.option("--glottocode", default="java1254", type=str, help="Single value or comma-separated list.")
@click.option("--iso6393", default=None, type=str)
@click.option("--inventory_id", default=None, type=str)
def main(phoible_csv_path, output_path, glottocode, iso6393, inventory_id):
    glottocodes = _parse_csv_option(glottocode)
    iso6393_codes = _parse_csv_option(iso6393)
    inventory_ids = _parse_csv_option(inventory_id)

    phoible_feature_dict = load_phoible_feature_dict(
        phoible_csv_path,
        glottocode=glottocodes,
        iso6393=iso6393_codes,
        inventory_id=inventory_ids,
    )
    feature_table, missing = build_lpep_phoible_table(
        token_to_id=token_to_id,
        phoible_feature_dict=phoible_feature_dict,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(
        {
            "feature_table": feature_table,
            "missing": missing,
            "token_to_id": token_to_id,
            "glottocode": glottocodes,
            "iso6393": iso6393_codes,
            "inventory_id": inventory_ids,
        },
        output_path,
    )

    print(f"Saved PHOIBLE LPEP table to {output_path}")
    print(f"Feature table shape: {tuple(feature_table.shape)}")
    print(f"Missing tokens ({len(missing)}): {missing}")
    print(f"Filters: glottocode={glottocodes}, iso6393={iso6393_codes}, inventory_id={inventory_ids}")

if __name__ == "__main__":
    main()
