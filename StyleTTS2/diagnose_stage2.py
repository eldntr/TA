import os.path as osp
import traceback

import click
import numpy as np
import torch
import yaml
from munch import Munch

from meldataset import build_dataloader
from models import (
    build_model,
    load_ASR_models,
    load_F0_models,
    load_checkpoint,
    maybe_apply_ppim,
    move_model_to_device,
)
from utils import (
    build_lpep_inputs,
    get_data_path_list,
    length_to_mask,
    load_lpep_feature_table,
    log_norm,
    maximum_path,
    mask_from_lens,
    recursive_munch,
)
from Utils.PLBERT.util import load_plbert


def summarize_tensor(name, value):
    if not torch.is_tensor(value):
        return {
            "name": name,
            "type": type(value).__name__,
            "value": value,
        }

    detached = value.detach()
    finite_mask = torch.isfinite(detached)
    finite_count = int(finite_mask.sum().item())
    total_count = detached.numel()
    summary = {
        "name": name,
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "finite": bool(torch.isfinite(detached).all().item()),
        "finite_count": finite_count,
        "total_count": total_count,
    }

    if finite_count > 0:
        finite_vals = detached[finite_mask]
        stats_vals = finite_vals.float() if not torch.is_floating_point(finite_vals) else finite_vals
        summary.update(
            {
                "min": float(finite_vals.min().item()),
                "max": float(finite_vals.max().item()),
                "mean": float(stats_vals.mean().item()),
                "std": float(stats_vals.std().item()) if stats_vals.numel() > 1 else 0.0,
            }
        )
    else:
        summary.update({"min": None, "max": None, "mean": None, "std": None})

    return summary


def print_summary(summary):
    parts = [f"[{summary['name']}]"]
    if "shape" in summary:
        parts.append(f"shape={summary['shape']}")
        parts.append(f"dtype={summary['dtype']}")
        parts.append(f"finite={summary['finite']}")
        parts.append(f"finite_count={summary['finite_count']}/{summary['total_count']}")
        parts.append(f"min={summary['min']}")
        parts.append(f"max={summary['max']}")
        parts.append(f"mean={summary['mean']}")
        parts.append(f"std={summary['std']}")
    else:
        parts.append(f"type={summary['type']}")
        parts.append(f"value={summary['value']}")
    print(" ".join(parts))


def require_finite(name, value):
    summary = summarize_tensor(name, value)
    print_summary(summary)
    if torch.is_tensor(value) and not summary["finite"]:
        raise RuntimeError(f"Non-finite tensor detected at {name}")
    return value


def load_batch(config, device):
    data_params = config["data_params"]
    train_list, _ = get_data_path_list(data_params["train_data"], data_params["val_data"])
    train_dataloader = build_dataloader(
        train_list,
        data_params["root_path"],
        OOD_data=data_params["OOD_data"],
        min_length=data_params["min_length"],
        batch_size=config.get("batch_size", 4),
        num_workers=0,
        device=device,
        dataset_config=data_params.get("dataset_config", {}),
    )
    batch = next(iter(train_dataloader))
    waves = batch[0]
    tensors = [b.to(device) for b in batch[1:]]
    texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels, lang_ids = tensors
    return waves, texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels, lang_ids


@click.command()
@click.option("-p", "--config_path", default="Configs/config_styletts2.yml", type=str)
@click.option("--device", default=None, type=str)
def main(config_path, device):
    config = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    device = device or config.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    print(f"Using config: {config_path}")
    print(f"Using device: {device}")

    waves, texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels, lang_ids = load_batch(config, device)
    print(f"Loaded batch with {len(waves)} items")
    print_summary(summarize_tensor("texts", texts))
    print_summary(summarize_tensor("mels", mels))
    print_summary(summarize_tensor("mel_input_length", mel_input_length))

    text_aligner = load_ASR_models(config.get("ASR_path"), config.get("ASR_config"))
    pitch_extractor = load_F0_models(config.get("F0_path"))
    plbert = load_plbert(config.get("PLBERT_dir"))

    model_params = recursive_munch(config["model_params"])
    phoible_feature_table = load_lpep_feature_table(model_params)
    use_lpep = getattr(model_params, "use_lpep", False)
    multispeaker = model_params.multispeaker

    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    move_model_to_device(model, device)
    n_down = model.text_aligner.n_down

    if config.get("pretrained_model", "") and config.get("second_stage_load_pretrained", False):
        checkpoint_path = config["pretrained_model"]
        print(f"Loading second-stage checkpoint: {checkpoint_path}")
        model, _, _, _ = load_checkpoint(
            model,
            None,
            checkpoint_path,
            load_only_params=config.get("load_only_params", True),
        )
    else:
        first_stage_path = config.get("first_stage_path", "")
        if not first_stage_path:
            raise RuntimeError("first_stage_path is required for diagnosis")
        checkpoint_path = osp.join(config["log_dir"], first_stage_path)
        print(f"Loading first-stage checkpoint: {checkpoint_path}")
        model, _, _, _ = load_checkpoint(
            model,
            None,
            checkpoint_path,
            load_only_params=True,
            ignore_modules=[
                "bert",
                "bert_encoder",
                "predictor",
                "predictor_encoder",
                "msd",
                "mpd",
                "wd",
                "diffusion",
            ],
        )
        model.predictor_encoder.load_state_dict(model.style_encoder.state_dict())

    for _, module in model.items():
        if isinstance(module, torch.nn.Module):
            module.eval()

    try:
        with torch.no_grad():
            mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
            mel_mask = length_to_mask(mel_input_length).to(device)
            text_mask = length_to_mask(input_lengths).to(device)
            require_finite("mask", mask.float())
            require_finite("mel_mask", mel_mask.float())
            require_finite("text_mask", text_mask.float())

            _, _, s2s_attn = model.text_aligner(mels, mask, texts)
            s2s_attn = s2s_attn.transpose(-1, -2)
            s2s_attn = s2s_attn[..., 1:]
            s2s_attn = s2s_attn.transpose(-1, -2)
            require_finite("s2s_attn", s2s_attn)

            mask_st = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
            s2s_attn_mono = maximum_path(s2s_attn, mask_st)
            require_finite("s2s_attn_mono", s2s_attn_mono)

            if use_lpep:
                lang_id, phon_feats = build_lpep_inputs(
                    texts,
                    model_params,
                    phoible_feature_table=phoible_feature_table,
                    lang_id=lang_ids,
                )
                t_en = model.text_encoder(
                    texts,
                    input_lengths,
                    text_mask,
                    lang_id=lang_id,
                    phon_feats=phon_feats,
                )
            else:
                t_en = model.text_encoder(texts, input_lengths, text_mask)
            require_finite("text_encoder", t_en)

            asr = t_en @ s2s_attn_mono
            d_gt = s2s_attn_mono.sum(axis=-1).detach()
            require_finite("asr", asr)
            require_finite("d_gt", d_gt)

            ss = []
            gs = []
            for batch_index in range(len(mel_input_length)):
                mel = mels[batch_index, :, : mel_input_length[batch_index]]
                ss.append(model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1)))
                gs.append(model.style_encoder(mel.unsqueeze(0).unsqueeze(1)))

            s_dur_global = torch.stack(ss).squeeze()
            gs = torch.stack(gs).squeeze()
            s_trg = torch.cat([gs, s_dur_global], dim=-1).detach()
            require_finite("s_dur_global", s_dur_global)
            require_finite("gs", gs)
            require_finite("s_trg", s_trg)

            bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
            require_finite("bert", bert_dur)
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
            d_en = maybe_apply_ppim(model, d_en, s_dur_global, text_mask)
            require_finite("bert_encoder", d_en)

            d, p = model.predictor(d_en, s_dur_global, input_lengths, s2s_attn_mono, text_mask)
            require_finite("predictor_d", d)
            require_finite("predictor_p", p)

            max_len = config.get("max_len", 200)
            mel_len = min(int(mel_input_length.min().item() / 2 - 1), max_len // 2)
            mel_len_st = int(mel_input_length.min().item() / 2 - 1)
            print(f"mel_len={mel_len} mel_len_st={mel_len_st}")
            if mel_len <= 0 or mel_len_st <= 0:
                raise RuntimeError(f"Invalid crop lengths: mel_len={mel_len}, mel_len_st={mel_len_st}")

            en = []
            gt = []
            st = []
            p_en = []
            wav = []
            starts = []

            for batch_index in range(len(mel_input_length)):
                mel_length = int(mel_input_length[batch_index].item() / 2)
                random_start = np.random.randint(0, mel_length - mel_len)
                starts.append((batch_index, random_start, mel_length))
                en.append(asr[batch_index, :, random_start : random_start + mel_len])
                p_en.append(p[batch_index, :, random_start : random_start + mel_len])
                gt.append(mels[batch_index, :, (random_start * 2) : ((random_start + mel_len) * 2)])
                clip = waves[batch_index][(random_start * 2) * 300 : ((random_start + mel_len) * 2) * 300]
                wav.append(torch.from_numpy(clip).to(device))

                random_start_st = np.random.randint(0, mel_length - mel_len_st)
                st.append(mels[batch_index, :, (random_start_st * 2) : ((random_start_st + mel_len_st) * 2)])

            print(f"crop_starts={starts}")

            wav = torch.stack(wav).float().detach()
            en = torch.stack(en)
            p_en = torch.stack(p_en)
            gt = torch.stack(gt).detach()
            st = torch.stack(st).detach()
            require_finite("en", en)
            require_finite("p_en", p_en)
            require_finite("gt", gt)
            require_finite("st", st)
            require_finite("wav", wav)

            s_dur = model.predictor_encoder(gt.unsqueeze(1) if multispeaker else gt.unsqueeze(1))
            s = model.style_encoder(st.unsqueeze(1) if multispeaker else gt.unsqueeze(1))
            require_finite("predictor_encoder_local", s_dur)
            require_finite("style_encoder_local", s)

            f0_real, _, f0 = model.pitch_extractor(gt.unsqueeze(1))
            f0 = f0.reshape(f0.shape[0], f0.shape[1] * 2, f0.shape[2], 1).squeeze()
            n_real = log_norm(gt.unsqueeze(1)).squeeze(1)
            require_finite("f0_real", f0_real)
            require_finite("f0_aux", f0)
            require_finite("n_real", n_real)

            y_rec_gt_pred = model.decoder(en, f0_real, n_real, s)
            require_finite("decoder_teacher_forced", y_rec_gt_pred)

            f0_fake, n_fake = model.predictor.F0Ntrain(p_en, s_dur)
            require_finite("f0_fake", f0_fake)
            require_finite("n_fake", n_fake)

            y_rec = model.decoder(en, f0_fake, n_fake, s)
            require_finite("decoder_predicted", y_rec)

            loss_f0_rec = torch.nn.functional.smooth_l1_loss(f0_real, f0_fake) / 10
            loss_norm_rec = torch.nn.functional.smooth_l1_loss(n_real, n_fake)
            require_finite("loss_f0_rec", loss_f0_rec)
            require_finite("loss_norm_rec", loss_norm_rec)

        print("Diagnosis completed without non-finite tensors in the tested forward path.")
    except Exception as exc:
        print(f"Diagnosis stopped: {exc}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
