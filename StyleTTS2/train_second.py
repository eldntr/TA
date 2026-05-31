# load packages
import random
import yaml
import time
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import click
import shutil
import traceback
import warnings
warnings.simplefilter('ignore')
from torch.utils.tensorboard import SummaryWriter

from meldataset import build_dataloader

from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet
from Utils.PLBERT.util import load_plbert

from models import *
from losses import *
from utils import *

from Modules.slmadv import SLMAdversarialLoss
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

from optimizers import build_optimizer

# simple fix for dataparallel that allows access to class attributes
class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
        
import logging
from logging import StreamHandler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


def summarize_tensor(name, value):
    if not torch.is_tensor(value):
        return {"name": name, "type": type(value).__name__, "value": value}

    data = value.detach()
    finite_mask = torch.isfinite(data)
    finite_count = int(finite_mask.sum().item())
    total_count = data.numel()
    summary = {
        "name": name,
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "device": str(data.device),
        "finite": bool(torch.isfinite(data).all().item()),
        "finite_count": finite_count,
        "total_count": total_count,
    }

    if finite_count > 0:
        finite_vals = data[finite_mask]
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


def summarize_module_state(name, module, attr):
    total_count = 0
    finite_count = 0
    max_abs = 0.0
    has_values = False

    for param in module.parameters():
        tensor = getattr(param, attr, None) if attr == "grad" else param.data
        if tensor is None:
            continue
        tensor = tensor.detach()
        total_count += tensor.numel()
        finite_mask = torch.isfinite(tensor)
        finite_count += int(finite_mask.sum().item())
        if finite_mask.any():
            has_values = True
            max_abs = max(max_abs, float(tensor[finite_mask].abs().max().item()))

    return {
        "name": name,
        "attr": attr,
        "finite": finite_count == total_count,
        "finite_count": finite_count,
        "total_count": total_count,
        "max_abs": max_abs if has_values else None,
    }


def save_debug_dump(log_dir, epoch, step, stage, summaries, raw_tensors=None):
    dump_path = osp.join(log_dir, f"nonfinite_debug_e{epoch + 1:04d}_s{step + 1:05d}.pt")
    payload = {
        "epoch": epoch + 1,
        "step": step + 1,
        "stage": stage,
        "summaries": summaries,
        "raw_tensors": {},
    }
    if raw_tensors:
        for key, value in raw_tensors.items():
            if torch.is_tensor(value):
                payload["raw_tensors"][key] = value.detach().cpu()
            else:
                payload["raw_tensors"][key] = value
    torch.save(payload, dump_path)
    logger.error("Saved non-finite debug dump to %s", dump_path)


def ensure_finite(name, value, summaries):
    summary = summarize_tensor(name, value)
    summaries[name] = summary
    if torch.is_tensor(value) and not summary["finite"]:
        raise FloatingPointError(f"Non-finite tensor at {name}")
    return value


def ensure_module_finite(name, module, attr, summaries):
    summary = summarize_module_state(name, module, attr)
    summaries[f"{name}_{attr}"] = summary
    if summary["total_count"] > 0 and not summary["finite"]:
        raise FloatingPointError(f"Non-finite module {attr} at {name}")
    return summary


def copy_matching_module_weights(dst_module, src_module):
    dst = getattr(dst_module, "module", dst_module)
    src = getattr(src_module, "module", src_module)
    dst.load_state_dict(src.state_dict(), strict=False)


def unwrap_module(module):
    return getattr(module, "module", module)


@click.command()
@click.option('-p', '--config_path', default='Configs/config.yml', type=str)
def main(config_path):
    config = yaml.safe_load(open(config_path))
    
    log_dir = config['log_dir']
    if not osp.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    writer = SummaryWriter(log_dir + "/tensorboard")

    # write logs
    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.addHandler(file_handler)

    
    batch_size = config.get('batch_size', 10)
    use_data_parallel = config.get('use_data_parallel', torch.cuda.device_count() > 1)

    epochs = config.get('epochs_2nd', 200)
    save_freq = config.get('save_freq', 2)
    log_interval = config.get('log_interval', 10)
    saving_epoch = config.get('save_freq', 2)
    diagnose_non_finite = False
    fail_fast_on_non_finite = False

    data_params = config.get('data_params', None)
    sr = config['preprocess_params'].get('sr', 24000)
    train_path = data_params['train_data']
    val_path = data_params['val_data']
    root_path = data_params['root_path']
    min_length = data_params['min_length']
    OOD_data = data_params['OOD_data']
    dataset_config = data_params.get('dataset_config', {})

    max_len = config.get('max_len', 200)
    
    loss_params = Munch(config['loss_params'])
    diff_epoch = loss_params.diff_epoch
    joint_epoch = loss_params.joint_epoch
    
    optimizer_params = Munch(config['optimizer_params'])
    
    train_list, val_list = get_data_path_list(train_path, val_path)
    device = 'cuda'

    train_dataloader = build_dataloader(train_list,
                                        root_path,
                                        OOD_data=OOD_data,
                                        min_length=min_length,
                                        batch_size=batch_size,
                                        num_workers=2,
                                        dataset_config=dataset_config,
                                        device=device)

    val_dataloader = build_dataloader(val_list,
                                      root_path,
                                      OOD_data=OOD_data,
                                      min_length=min_length,
                                      batch_size=batch_size,
                                      validation=True,
                                      num_workers=0,
                                      device=device,
                                      dataset_config=dataset_config)
    
    # load pretrained ASR model
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)
    
    # load pretrained F0 model
    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)
    
    # load PL-BERT model
    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(BERT_path)
    
    # build model
    model_params = recursive_munch(config['model_params'])
    multispeaker = model_params.multispeaker
    phoible_feature_table = load_lpep_feature_table(model_params)
    use_lpep = getattr(model_params, "use_lpep", False)
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    move_model_to_device(model, device)
    
    # DataParallel is unstable here on some single-GPU setups, especially with spectral_norm modules.
    if use_data_parallel:
        for key in model:
            if key != "mpd" and key != "msd" and key != "wd" and isinstance(model[key], nn.Module):
                model[key] = MyDataParallel(model[key])
    logger.info("use_data_parallel=%s", use_data_parallel)
            
    start_epoch = 0
    iters = 0

    load_pretrained = config.get('pretrained_model', '') != '' and config.get('second_stage_load_pretrained', False)
    
    if not load_pretrained:
        if config.get('first_stage_path', '') != '':
            first_stage_path = osp.join(log_dir, config.get('first_stage_path', 'first_stage.pth'))
            print('Loading the first stage model at %s ...' % first_stage_path)
            model, _, start_epoch, iters = load_checkpoint(model, 
                None, 
                first_stage_path,
                load_only_params=True,
                ignore_modules=['bert', 'bert_encoder', 'predictor', 'predictor_encoder', 'msd', 'mpd', 'wd', 'diffusion']) # keep starting epoch for tensorboard log

            # these epochs should be counted from the start epoch
            diff_epoch += start_epoch
            joint_epoch += start_epoch
            epochs += start_epoch
            
            copy_matching_module_weights(model.predictor_encoder, model.style_encoder)
        else:
            raise ValueError('You need to specify the path to the first stage model.') 

    gl = GeneratorLoss(model.mpd, model.msd).to(device)
    dl = DiscriminatorLoss(model.mpd, model.msd).to(device)
    wl = WavLMLoss(model_params.slm.model, 
                   model.wd, 
                   sr, 
                   model_params.slm.sr).to(device)

    if use_data_parallel:
        gl = MyDataParallel(gl)
        dl = MyDataParallel(dl)
        wl = MyDataParallel(wl)
    
    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
        clamp=False
    )
    
    scheduler_params = {
        "max_lr": optimizer_params.lr,
        "pct_start": float(0),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }
    model_keys = [key for key, _ in model_module_items(model)]
    scheduler_params_dict= {key: scheduler_params.copy() for key in model_keys}
    scheduler_params_dict['bert']['max_lr'] = optimizer_params.bert_lr * 2
    scheduler_params_dict['decoder']['max_lr'] = optimizer_params.ft_lr * 2
    scheduler_params_dict['style_encoder']['max_lr'] = optimizer_params.ft_lr * 2
    
    optimizer = build_optimizer({key: model[key].parameters() for key in model_keys},
                                          scheduler_params_dict=scheduler_params_dict, lr=optimizer_params.lr)
    
    # adjust BERT learning rate
    for g in optimizer.optimizers['bert'].param_groups:
        g['betas'] = (0.9, 0.99)
        g['lr'] = optimizer_params.bert_lr
        g['initial_lr'] = optimizer_params.bert_lr
        g['min_lr'] = 0
        g['weight_decay'] = 0.01
        
    # adjust acoustic module learning rate
    for module in ["decoder", "style_encoder"]:
        for g in optimizer.optimizers[module].param_groups:
            g['betas'] = (0.0, 0.99)
            g['lr'] = optimizer_params.ft_lr
            g['initial_lr'] = optimizer_params.ft_lr
            g['min_lr'] = 0
            g['weight_decay'] = 1e-4
        
    # load models if there is a model
    if load_pretrained:
        model, optimizer, start_epoch, iters = load_checkpoint(
            model, optimizer, config['pretrained_model'],
            load_only_params=config.get('load_only_params', True),
            ignore_modules=['bert', 'bert_encoder', 'text_encoder', 'text_aligner']
        )
        
    n_down = model.text_aligner.n_down

    best_loss = float('inf')  # best test loss
    loss_train_record = list([])
    loss_test_record = list([])
    iters = 0
    
    criterion = nn.L1Loss() # F0 loss (regression)
    torch.cuda.empty_cache()
    
    stft_loss = MultiResolutionSTFTLoss().to(device)
    
    print('BERT', optimizer.optimizers['bert'])
    print('decoder', optimizer.optimizers['decoder'])

    start_ds = False
    
    running_std = []
    
    slmadv_params = Munch(config['slmadv_params'])
    slmadv = SLMAdversarialLoss(model, wl, sampler, 
                                slmadv_params.min_len, 
                                slmadv_params.max_len,
                                batch_percentage=slmadv_params.batch_percentage,
                                skip_update=slmadv_params.iter, 
                                sig=slmadv_params.sig,
                                lpep_args=model_params,
                                phoible_feature_table=phoible_feature_table,
                               )


    for epoch in range(start_epoch, epochs):
        running_loss = 0
        start_time = time.time()

        set_model_mode(model, train=False)

        model.predictor.train()
        model.bert_encoder.train()
        model.bert.train()
        model.msd.train()
        model.mpd.train()
        if getattr(model, "ppim", None) is not None:
            model.ppim.train()


        if epoch >= diff_epoch:
            start_ds = True

        for i, batch in enumerate(train_dataloader):
            waves = batch[0]
            batch = [b.to(device) for b in batch[1:]]
            texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels, lang_ids = batch
            debug_summaries = {}
            debug_raw_tensors = {}

            with torch.no_grad():
                mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                mel_mask = length_to_mask(mel_input_length).to(device)
                text_mask = length_to_mask(input_lengths).to(texts.device)

                try:
                    _, _, s2s_attn = model.text_aligner(mels, mask, texts)
                    s2s_attn = s2s_attn.transpose(-1, -2)
                    s2s_attn = s2s_attn[..., 1:]
                    s2s_attn = s2s_attn.transpose(-1, -2)
                except:
                    continue

                mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
                s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

                # encode
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
                asr = (t_en @ s2s_attn_mono)

                d_gt = s2s_attn_mono.sum(axis=-1).detach()
                
                # compute reference styles
                if multispeaker and epoch >= diff_epoch:
                    ref_ss = model.style_encoder(ref_mels.unsqueeze(1))
                    ref_sp = model.predictor_encoder(ref_mels.unsqueeze(1))
                    ref = torch.cat([ref_ss, ref_sp], dim=1)

            # compute the style of the entire utterance
            # this operation cannot be done in batch because of the avgpool layer (may need to work on masked avgpool)
            ss = []
            gs = []
            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item())
                mel = mels[bib, :, :mel_input_length[bib]]
                s = model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1))
                ss.append(s)
                s = model.style_encoder(mel.unsqueeze(0).unsqueeze(1))
                gs.append(s)

            s_dur = torch.stack(ss).squeeze()  # global prosodic styles
            gs = torch.stack(gs).squeeze() # global acoustic styles
            s_trg = torch.cat([gs, s_dur], dim=-1).detach() # ground truth for denoiser

            bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 
            if getattr(model_params, "use_ppim", False):
                d_en = maybe_apply_ppim(model, d_en, s_dur, text_mask)
            
            # denoiser training
            if epoch >= diff_epoch:
                num_steps = np.random.randint(3, 5)
                diffusion_module = unwrap_module(model.diffusion)
                
                if model_params.diffusion.dist.estimate_sigma_data:
                    diffusion_module.diffusion.sigma_data = s_trg.std(axis=-1).mean().item() # batch-wise std estimation
                    running_std.append(diffusion_module.diffusion.sigma_data)
                    
                if multispeaker:
                    s_preds = sampler(noise = torch.randn_like(s_trg).unsqueeze(1).to(device), 
                          embedding=bert_dur,
                          embedding_scale=1,
                                   features=ref, # reference from the same speaker as the embedding
                             embedding_mask_proba=0.1,
                             num_steps=num_steps).squeeze(1)
                    loss_diff = model.diffusion(s_trg.unsqueeze(1), embedding=bert_dur, features=ref).mean() # EDM loss
                    loss_sty = F.l1_loss(s_preds, s_trg.detach()) # style reconstruction loss
                else:
                    s_preds = sampler(noise = torch.randn_like(s_trg).unsqueeze(1).to(device), 
                          embedding=bert_dur,
                          embedding_scale=1,
                             embedding_mask_proba=0.1,
                             num_steps=num_steps).squeeze(1)                    
                    loss_diff = diffusion_module.diffusion(s_trg.unsqueeze(1), embedding=bert_dur).mean() # EDM loss
                    loss_sty = F.l1_loss(s_preds, s_trg.detach()) # style reconstruction loss
            else:
                loss_sty = 0
                loss_diff = 0

            d, p = model.predictor(d_en, s_dur, 
                                                    input_lengths, 
                                                    s2s_attn_mono, 
                                                    text_mask)
            
            mel_len = min(int(mel_input_length.min().item() / 2 - 1), max_len // 2)
            mel_len_st = int(mel_input_length.min().item() / 2 - 1)
            en = []
            gt = []
            st = []
            p_en = []
            wav = []

            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item() / 2)

                random_start = np.random.randint(0, mel_length - mel_len)
                en.append(asr[bib, :, random_start:random_start+mel_len])
                p_en.append(p[bib, :, random_start:random_start+mel_len])
                gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])
                
                y = waves[bib][(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
                wav.append(torch.from_numpy(y).to(device))

                # style reference (better to be different from the GT)
                random_start = np.random.randint(0, mel_length - mel_len_st)
                st.append(mels[bib, :, (random_start * 2):((random_start+mel_len_st) * 2)])
                
            wav = torch.stack(wav).float().detach()

            en = torch.stack(en)
            p_en = torch.stack(p_en)
            gt = torch.stack(gt).detach()
            st = torch.stack(st).detach()
            
            if gt.size(-1) < 80:
                continue

            s_dur = model.predictor_encoder(st.unsqueeze(1) if multispeaker else gt.unsqueeze(1))
            s = model.style_encoder(st.unsqueeze(1) if multispeaker else gt.unsqueeze(1))
            
            try:
                if diagnose_non_finite:
                    debug_raw_tensors.update({
                        "texts": texts,
                        "input_lengths": input_lengths,
                        "mel_input_length": mel_input_length,
                        "en": en,
                        "p_en": p_en,
                        "gt": gt,
                        "st": st,
                        "wav": wav,
                    })
                    ensure_finite("s2s_attn_mono", s2s_attn_mono, debug_summaries)
                    ensure_finite("asr", asr, debug_summaries)
                    ensure_finite("d_gt", d_gt, debug_summaries)
                    ensure_finite("s_dur", s_dur, debug_summaries)
                    ensure_finite("style", s, debug_summaries)
                    ensure_finite("predictor_d", d, debug_summaries)
                    ensure_finite("predictor_p", p, debug_summaries)

                with torch.no_grad():
                    F0_real, _, F0 = model.pitch_extractor(gt.unsqueeze(1))
                    F0 = F0.reshape(F0.shape[0], F0.shape[1] * 2, F0.shape[2], 1).squeeze()

                    asr_real = model.text_aligner.get_feature(gt)

                    N_real = log_norm(gt.unsqueeze(1)).squeeze(1)
                    
                    y_rec_gt = wav.unsqueeze(1)
                    y_rec_gt_pred = model.decoder(en, F0_real, N_real, s)

                    if epoch >= joint_epoch:
                        # ground truth from recording
                        wav = y_rec_gt # use recording since decoder is tuned
                    else:
                        # ground truth from reconstruction
                        wav = y_rec_gt_pred # use reconstruction since decoder is fixed

                if diagnose_non_finite:
                    debug_raw_tensors.update({
                        "F0_real": F0_real,
                        "N_real": N_real,
                        "y_rec_gt_pred": y_rec_gt_pred,
                    })
                    ensure_finite("F0_real", F0_real, debug_summaries)
                    ensure_finite("F0_aux", F0, debug_summaries)
                    ensure_finite("asr_real", asr_real, debug_summaries)
                    ensure_finite("N_real", N_real, debug_summaries)
                    ensure_finite("y_rec_gt_pred", y_rec_gt_pred, debug_summaries)
                    ensure_finite("wav_target", wav, debug_summaries)

                F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s_dur)
                y_rec = model.decoder(en, F0_fake, N_fake, s)

                debug_raw_tensors.update({
                    "F0_fake": F0_fake,
                    "N_fake": N_fake,
                    "y_rec": y_rec,
                })
                ensure_finite("F0_fake", F0_fake, debug_summaries)
                ensure_finite("N_fake", N_fake, debug_summaries)
                ensure_finite("y_rec", y_rec, debug_summaries)

                loss_F0_rec =  (F.smooth_l1_loss(F0_real, F0_fake)) / 10
                loss_norm_rec = F.smooth_l1_loss(N_real, N_fake)
                ensure_finite("loss_F0_rec", loss_F0_rec, debug_summaries)
                ensure_finite("loss_norm_rec", loss_norm_rec, debug_summaries)

                if start_ds:
                    optimizer.zero_grad()
                    d_loss = dl(wav.detach(), y_rec.detach()).mean()
                    ensure_finite("d_loss", d_loss, debug_summaries)
                    d_loss.backward()
                    if diagnose_non_finite:
                        ensure_module_finite("msd", model.msd, "grad", debug_summaries)
                        ensure_module_finite("mpd", model.mpd, "grad", debug_summaries)
                    optimizer.step('msd')
                    optimizer.step('mpd')
                else:
                    d_loss = 0

                # generator loss
                optimizer.zero_grad()

                loss_mel = stft_loss(y_rec, wav)
                if start_ds:
                    loss_gen_all = gl(wav, y_rec).mean()
                else:
                    loss_gen_all = 0
                loss_lm = wl(wav.detach().squeeze(), y_rec.squeeze()).mean()

                ensure_finite("loss_mel", loss_mel, debug_summaries)
                if torch.is_tensor(loss_gen_all):
                    ensure_finite("loss_gen_all", loss_gen_all, debug_summaries)
                ensure_finite("loss_lm", loss_lm, debug_summaries)

                loss_ce = 0
                loss_dur = 0
                valid_dur_items = 0
                for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths):
                    _text_length = int(_text_length.item()) if torch.is_tensor(_text_length) else int(_text_length)
                    _s2s_pred = _s2s_pred[:_text_length, :]
                    _text_input = _text_input[:_text_length].long()
                    _s2s_trg = torch.zeros_like(_s2s_pred)
                    for p in range(_s2s_trg.shape[0]):
                        _s2s_trg[p, :_text_input[p]] = 1
                    _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)

                    if _text_length > 2:
                        loss_dur += F.l1_loss(
                            _dur_pred[1:_text_length-1],
                            _text_input[1:_text_length-1],
                        )
                        valid_dur_items += 1
                    loss_ce += F.binary_cross_entropy_with_logits(_s2s_pred.flatten(), _s2s_trg.flatten())

                loss_ce /= texts.size(0)
                if valid_dur_items > 0:
                    loss_dur /= valid_dur_items
                else:
                    loss_dur = torch.zeros((), device=texts.device)

                ensure_finite("loss_ce", loss_ce, debug_summaries)
                ensure_finite("loss_dur", loss_dur, debug_summaries)

                g_loss = loss_params.lambda_mel * loss_mel + \
                         loss_params.lambda_F0 * loss_F0_rec + \
                         loss_params.lambda_ce * loss_ce + \
                         loss_params.lambda_norm * loss_norm_rec + \
                         loss_params.lambda_dur * loss_dur + \
                         loss_params.lambda_gen * loss_gen_all + \
                         loss_params.lambda_slm * loss_lm + \
                         loss_params.lambda_sty * loss_sty + \
                         loss_params.lambda_diff * loss_diff

                running_loss += loss_mel.item()
                ensure_finite("loss_sty", loss_sty if torch.is_tensor(loss_sty) else torch.tensor(loss_sty, device=texts.device), debug_summaries)
                ensure_finite("loss_diff", loss_diff if torch.is_tensor(loss_diff) else torch.tensor(loss_diff, device=texts.device), debug_summaries)
                ensure_finite("g_loss", g_loss, debug_summaries)
                g_loss.backward()

                if diagnose_non_finite:
                    ensure_module_finite("bert_encoder", model.bert_encoder, "grad", debug_summaries)
                    ensure_module_finite("bert", model.bert, "grad", debug_summaries)
                    ensure_module_finite("predictor", model.predictor, "grad", debug_summaries)
                    ensure_module_finite("predictor_encoder", model.predictor_encoder, "grad", debug_summaries)
                    if epoch >= diff_epoch:
                        ensure_module_finite("diffusion", model.diffusion, "grad", debug_summaries)
                    if epoch >= joint_epoch:
                        ensure_module_finite("style_encoder", model.style_encoder, "grad", debug_summaries)
                        ensure_module_finite("decoder", model.decoder, "grad", debug_summaries)

                optimizer.step('bert_encoder')
                optimizer.step('bert')
                optimizer.step('predictor')
                if 'ppim' in optimizer.optimizers:
                    optimizer.step('ppim')
                optimizer.step('predictor_encoder')
                
                if epoch >= diff_epoch:
                    optimizer.step('diffusion')
                
                if diagnose_non_finite:
                    ensure_module_finite("bert_encoder", model.bert_encoder, "data", debug_summaries)
                    ensure_module_finite("bert", model.bert, "data", debug_summaries)
                    ensure_module_finite("predictor", model.predictor, "data", debug_summaries)
                    ensure_module_finite("predictor_encoder", model.predictor_encoder, "data", debug_summaries)
                    if epoch >= diff_epoch:
                        ensure_module_finite("diffusion", model.diffusion, "data", debug_summaries)

                if epoch >= joint_epoch:
                    optimizer.step('style_encoder')
                    optimizer.step('decoder')
                    if diagnose_non_finite:
                        ensure_module_finite("style_encoder", model.style_encoder, "data", debug_summaries)
                        ensure_module_finite("decoder", model.decoder, "data", debug_summaries)
            except FloatingPointError as exc:
                logger.error("Non-finite detected at epoch %s step %s: %s", epoch + 1, i + 1, exc)
                save_debug_dump(log_dir, epoch, i, str(exc), debug_summaries, debug_raw_tensors)
                if fail_fast_on_non_finite:
                    raise RuntimeError(
                        f"Non-finite detected at epoch {epoch + 1} step {i + 1}: {exc}. "
                        f"See debug dump under {log_dir}."
                    ) from exc
                continue

            if epoch >= joint_epoch:
                # randomly pick whether to use in-distribution text
                if np.random.rand() < 0.5:
                    use_ind = True
                else:
                    use_ind = False

                if use_ind:
                    ref_lengths = input_lengths
                    ref_texts = texts
                    
                slm_out = slmadv(i, 
                                 y_rec_gt, 
                                 y_rec_gt_pred, 
                                 waves, 
                                 mel_input_length,
                                 ref_texts, 
                                 ref_lengths, use_ind, s_trg.detach(), ref if multispeaker else None, ref_lang_id=lang_ids)

                if slm_out is None:
                    continue
                    
                d_loss_slm, loss_gen_lm, y_pred = slm_out
                
                # SLM generator loss
                optimizer.zero_grad()
                loss_gen_lm.backward()

                # compute the gradient norm
                total_norm = {}
                for key, module in model_module_items(model):
                    total_norm[key] = 0
                    parameters = [p for p in module.parameters() if p.grad is not None and p.requires_grad]
                    for p in parameters:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm[key] += param_norm.item() ** 2
                    total_norm[key] = total_norm[key] ** 0.5

                # gradient scaling
                if total_norm['predictor'] > slmadv_params.thresh:
                    for _, module in model_module_items(model):
                        for p in module.parameters():
                            if p.grad is not None:
                                p.grad *= (1 / total_norm['predictor']) 

                for p in model.predictor.duration_proj.parameters():
                    if p.grad is not None:
                        p.grad *= slmadv_params.scale

                for p in model.predictor.lstm.parameters():
                    if p.grad is not None:
                        p.grad *= slmadv_params.scale

                for p in model.diffusion.parameters():
                    if p.grad is not None:
                        p.grad *= slmadv_params.scale

                optimizer.step('bert_encoder')
                optimizer.step('bert')
                optimizer.step('predictor')
                if 'ppim' in optimizer.optimizers:
                    optimizer.step('ppim')
                optimizer.step('diffusion')

                # SLM discriminator loss
                if d_loss_slm != 0:
                    optimizer.zero_grad()
                    d_loss_slm.backward(retain_graph=True)
                    optimizer.step('wd')

            else:
                d_loss_slm, loss_gen_lm = 0, 0
                
            iters = iters + 1
            
            if (i+1)%log_interval == 0:
                logger.info ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f, Disc Loss: %.5f, Dur Loss: %.5f, CE Loss: %.5f, Norm Loss: %.5f, F0 Loss: %.5f, LM Loss: %.5f, Gen Loss: %.5f, Sty Loss: %.5f, Diff Loss: %.5f, DiscLM Loss: %.5f, GenLM Loss: %.5f'
                    %(epoch+1, epochs, i+1, len(train_list)//batch_size, running_loss / log_interval, d_loss, loss_dur, loss_ce, loss_norm_rec, loss_F0_rec, loss_lm, loss_gen_all, loss_sty, loss_diff, d_loss_slm, loss_gen_lm))
                
                writer.add_scalar('train/mel_loss', running_loss / log_interval, iters)
                writer.add_scalar('train/gen_loss', loss_gen_all, iters)
                writer.add_scalar('train/d_loss', d_loss, iters)
                writer.add_scalar('train/ce_loss', loss_ce, iters)
                writer.add_scalar('train/dur_loss', loss_dur, iters)
                writer.add_scalar('train/slm_loss', loss_lm, iters)
                writer.add_scalar('train/norm_loss', loss_norm_rec, iters)
                writer.add_scalar('train/F0_loss', loss_F0_rec, iters)
                writer.add_scalar('train/sty_loss', loss_sty, iters)
                writer.add_scalar('train/diff_loss', loss_diff, iters)
                writer.add_scalar('train/d_loss_slm', d_loss_slm, iters)
                writer.add_scalar('train/gen_loss_slm', loss_gen_lm, iters)
                
                running_loss = 0
                
                print('Time elasped:', time.time()-start_time)
                
        loss_test = 0
        loss_align = 0
        loss_f = 0
        set_model_mode(model, train=False)

        with torch.no_grad():
            iters_test = 0
            for batch_idx, batch in enumerate(val_dataloader):
                optimizer.zero_grad()
                
                try:
                    waves = batch[0]
                    batch = [b.to(device) for b in batch[1:]]
                    texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels, lang_ids = batch
                    if texts.size(0) < 2 and torch.cuda.device_count() > 1:
                        continue
                    with torch.no_grad():
                        mask = length_to_mask(mel_input_length // (2 ** n_down)).to('cuda')
                        text_mask = length_to_mask(input_lengths).to(texts.device)

                        _, _, s2s_attn = model.text_aligner(mels, mask, texts)
                        s2s_attn = s2s_attn.transpose(-1, -2)
                        s2s_attn = s2s_attn[..., 1:]
                        s2s_attn = s2s_attn.transpose(-1, -2)

                        mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
                        s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

                        # encode
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
                        asr = (t_en @ s2s_attn_mono)

                        d_gt = s2s_attn_mono.sum(axis=-1).detach()

                    ss = []
                    gs = []

                    for bib in range(len(mel_input_length)):
                        mel_length = int(mel_input_length[bib].item())
                        mel = mels[bib, :, :mel_input_length[bib]]
                        s = model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1))
                        ss.append(s)
                        s = model.style_encoder(mel.unsqueeze(0).unsqueeze(1))
                        gs.append(s)

                    s = torch.stack(ss).squeeze()
                    gs = torch.stack(gs).squeeze()
                    s_trg = torch.cat([s, gs], dim=-1).detach()

                    bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
                    d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 
                    if getattr(model_params, "use_ppim", False):
                        d_en = maybe_apply_ppim(model, d_en, s, text_mask)
                    d, p = model.predictor(d_en, s, 
                                                        input_lengths, 
                                                        s2s_attn_mono, 
                                                        text_mask)
                    # get clips
                    mel_len = int(mel_input_length.min().item() / 2 - 1)
                    en = []
                    gt = []
                    p_en = []
                    wav = []

                    for bib in range(len(mel_input_length)):
                        mel_length = int(mel_input_length[bib].item() / 2)

                        random_start = np.random.randint(0, mel_length - mel_len)
                        en.append(asr[bib, :, random_start:random_start+mel_len])
                        p_en.append(p[bib, :, random_start:random_start+mel_len])

                        gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])

                        y = waves[bib][(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
                        wav.append(torch.from_numpy(y).to(device))

                    wav = torch.stack(wav).float().detach()

                    en = torch.stack(en)
                    p_en = torch.stack(p_en)
                    gt = torch.stack(gt).detach()

                    s = model.predictor_encoder(gt.unsqueeze(1))

                    F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s)

                    loss_dur = 0
                    valid_dur_items = 0
                    for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths):
                        _text_length = int(_text_length.item()) if torch.is_tensor(_text_length) else int(_text_length)
                        _s2s_pred = _s2s_pred[:_text_length, :]
                        _text_input = _text_input[:_text_length].long()
                        _s2s_trg = torch.zeros_like(_s2s_pred)
                        for bib in range(_s2s_trg.shape[0]):
                            _s2s_trg[bib, :_text_input[bib]] = 1
                        _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)
                        if _text_length > 2:
                            loss_dur += F.l1_loss(
                                _dur_pred[1:_text_length-1],
                                _text_input[1:_text_length-1],
                            )
                            valid_dur_items += 1

                    if valid_dur_items > 0:
                        loss_dur /= valid_dur_items
                    else:
                        loss_dur = torch.zeros((), device=texts.device)

                    s = model.style_encoder(gt.unsqueeze(1))

                    y_rec = model.decoder(en, F0_fake, N_fake, s)
                    loss_mel = stft_loss(y_rec.squeeze(), wav.detach())

                    F0_real, _, F0 = model.pitch_extractor(gt.unsqueeze(1)) 

                    loss_F0 = F.l1_loss(F0_real, F0_fake) / 10

                    loss_test += (loss_mel).mean()
                    loss_align += (loss_dur).mean()
                    loss_f += (loss_F0).mean()

                    iters_test += 1
                except Exception as e:
                    print(f"run into exception", e)
                    traceback.print_exc()
                    continue

        print('Epochs:', epoch + 1)
        logger.info('Validation loss: %.3f, Dur loss: %.3f, F0 loss: %.3f' % (loss_test / iters_test, loss_align / iters_test, loss_f / iters_test) + '\n\n\n')
        print('\n\n\n')
        writer.add_scalar('eval/mel_loss', loss_test / iters_test, epoch + 1)
        writer.add_scalar('eval/dur_loss', loss_align / iters_test, epoch + 1)
        writer.add_scalar('eval/F0_loss', loss_f / iters_test, epoch + 1)
        
        if epoch < joint_epoch:
            # generating reconstruction examples with GT duration
            
            with torch.no_grad():
                for bib in range(len(asr)):
                    mel_length = int(mel_input_length[bib].item())
                    gt = mels[bib, :, :mel_length].unsqueeze(0)
                    en = asr[bib, :, :mel_length // 2].unsqueeze(0)

                    F0_real, _, _ = model.pitch_extractor(gt.unsqueeze(1))
                    F0_real = F0_real.unsqueeze(0)
                    s = model.style_encoder(gt.unsqueeze(1))
                    real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)

                    y_rec = model.decoder(en, F0_real, real_norm, s)

                    writer.add_audio('eval/y' + str(bib), y_rec.cpu().numpy().squeeze(), epoch, sample_rate=sr)

                    s_dur = model.predictor_encoder(gt.unsqueeze(1))
                    p_en = p[bib, :, :mel_length // 2].unsqueeze(0)

                    F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s_dur)

                    y_pred = model.decoder(en, F0_fake, N_fake, s)

                    writer.add_audio('pred/y' + str(bib), y_pred.cpu().numpy().squeeze(), epoch, sample_rate=sr)

                    if epoch == 0:
                        writer.add_audio('gt/y' + str(bib), waves[bib].squeeze(), epoch, sample_rate=sr)

                    if bib >= 5:
                        break
        else:
            # generating sampled speech from text directly
            with torch.no_grad():
                # compute reference styles
                if multispeaker and epoch >= diff_epoch:
                    ref_ss = model.style_encoder(ref_mels.unsqueeze(1))
                    ref_sp = model.predictor_encoder(ref_mels.unsqueeze(1))
                    ref_s = torch.cat([ref_ss, ref_sp], dim=1)
                    
                for bib in range(len(d_en)):
                    if multispeaker:
                        s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(texts.device), 
                              embedding=bert_dur[bib].unsqueeze(0),
                              embedding_scale=1,
                                features=ref_s[bib].unsqueeze(0), # reference from the same speaker as the embedding
                                 num_steps=5).squeeze(1)
                    else:
                        s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(texts.device), 
                              embedding=bert_dur[bib].unsqueeze(0),
                              embedding_scale=1,
                                 num_steps=5).squeeze(1)

                    s = s_pred[:, 128:]
                    ref = s_pred[:, :128]

                    d = model.predictor.text_encoder(d_en[bib, :, :input_lengths[bib]].unsqueeze(0), 
                                                     s, input_lengths[bib, ...].unsqueeze(0), text_mask[bib, :input_lengths[bib]].unsqueeze(0))

                    x, _ = model.predictor.lstm(d)
                    duration = model.predictor.duration_proj(x)

                    duration = torch.sigmoid(duration).sum(axis=-1)
                    pred_dur = torch.round(duration.squeeze()).clamp(min=1)

                    pred_dur[-1] += 5

                    pred_aln_trg = torch.zeros(input_lengths[bib], int(pred_dur.sum().data))
                    c_frame = 0
                    for i in range(pred_aln_trg.size(0)):
                        pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                        c_frame += int(pred_dur[i].data)

                    # encode prosody
                    en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(texts.device))
                    F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
                    out = model.decoder((t_en[bib, :, :input_lengths[bib]].unsqueeze(0) @ pred_aln_trg.unsqueeze(0).to(texts.device)), 
                                            F0_pred, N_pred, ref.squeeze().unsqueeze(0))

                    writer.add_audio('pred/y' + str(bib), out.cpu().numpy().squeeze(), epoch, sample_rate=sr)

                    if bib >= 5:
                        break
                            
        if epoch % saving_epoch == 0:
            if (loss_test / iters_test) < best_loss:
                best_loss = loss_test / iters_test
            print('Saving..')
            state = {
                'net': model_state_dict(model),
                'optimizer': optimizer.state_dict(),
                'iters': iters,
                'val_loss': loss_test / iters_test,
                'epoch': epoch,
            }
            save_path = osp.join(log_dir, 'epoch_2nd_%05d.pth' % epoch)
            torch.save(state, save_path)
            
            # if estimate sigma, save the estimated simga
            if model_params.diffusion.dist.estimate_sigma_data:
                config['model_params']['diffusion']['dist']['sigma_data'] = float(np.mean(running_std))
                
                with open(osp.join(log_dir, osp.basename(config_path)), 'w') as outfile:
                    yaml.dump(config, outfile, default_flow_style=True)
        
if __name__=="__main__":
    main()
