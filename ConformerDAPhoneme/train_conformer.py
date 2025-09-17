import argparse
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import *
from data_load import *
import logging
import scoring
import compute_eer
from sklearn.metrics import balanced_accuracy_score
from torch.nn.functional import pairwise_distance
from data_load import PairedDataset, collate_fn_paired
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def validation(valid_txt, model, model_name, device, kaldi, num_lang, msg):
    valid_set = RawFeatures(valid_txt)
    valid_data = DataLoader(dataset=valid_set,
                            batch_size=1,
                            pin_memory=True,
                            shuffle=False,
                            collate_fn=collate_fn_atten)
    model.eval()
    correct = 0
    total = 0
    scores = 0
    with torch.no_grad():
        for step, (utt, labels, seq_len) in enumerate(valid_data):
            utt = utt.to(device=device, dtype=torch.float)
            labels = labels.to(device)
            atten_mask = get_atten_mask(seq_len, utt.size(0))
            atten_mask = atten_mask.to(device=device)
            # Forward pass
            outputs = model(utt, atten_mask)
            predicted = torch.argmax(outputs, -1)
            total += labels.size(-1)
            correct += (predicted == labels).sum().item()
            if step == 0:
                scores = outputs
            else: 
                scores = torch.cat((scores, outputs), dim=0)
    acc = correct / total
    logging.info('Current Acc.: {:.4f} %'.format(100 * acc))
    # for balanced Acc.
    prediction_all = torch.argmax(scores, -1).squeeze().cpu().numpy()
    with open(valid_txt, 'r') as f:
        lines = f.readlines()
    labels_array = np.array([int(x.split()[-1].strip()) for x in lines])

    scores = scores.squeeze().cpu().numpy()
    logging.info(f"{scores.shape}")
    trial_txt = 'trial_{}.txt'.format(model_name)
    score_txt = 'score_{}.txt'.format(model_name)
    scoring.get_trials(valid_txt, num_lang, trial_txt)
    scoring.get_score(valid_txt, scores, num_lang, score_txt)
    eer_txt = trial_txt.replace('trial', 'eer')
    target_score, non_target_score, score_all = compute_eer.load_file(score_txt, trial_txt)
    p_miss, p_fa = compute_eer.compute_rocch(target_score, non_target_score)
    eer = compute_eer.rocch2eer(p_miss, p_fa)
    cavg = scoring.compute_cavg(trial_txt, score_txt)
    logging.info(f"Currently processing {msg} files")
    logging.info(f"EER: {eer}")
    logging.info(f"Cavg: {cavg}")
    logging.info(f"Balanced Acc.: {balanced_accuracy_score(labels_array, prediction_all)}")

    return cavg


def main():
    parser = argparse.ArgumentParser(description='paras for training')
    parser.add_argument('--dim', type=int, help='dim of input features',
                        default=392)
    parser.add_argument('--model', type=str, help='model name',
                        default='Transformer')
    parser.add_argument('--train', type=str, help='training data, in .txt (source domain)')
    parser.add_argument('--test', type=str, help='testing data, in .txt (target domain)')
    parser.add_argument('--valid', type=str, help='validating data in .txt (source domain)')
    parser.add_argument('--batch', type=int, help='batch size',
                        default=8)
    parser.add_argument('--warmup', type=int, help='num of epochs',
                        default=12000)
    parser.add_argument('--epochs', type=int, help='num of epochs',
                        default=50)
    parser.add_argument('--lang', type=int, help='num of language classes',
                        default=12)  # Updated to 12 languages
    parser.add_argument('--lr', type=float, help='initial learning rate',
                        default=0.0001)
    parser.add_argument('--device', type=int, help='Device name',
                        default=0)
    parser.add_argument('--kaldi', type=str, help='kaldi root', default='/home/hexin/Desktop/kaldi/')
    parser.add_argument('--seed', type=int, help='Device name',
                        default=0)
    parser.add_argument('--accum_steps', type=int, help='gradient accumulation steps', default=1)
    parser.add_argument('--use_amp', action='store_true', help='use mixed precision training', default=False)

    args = parser.parse_args()

    setup_seed(args.seed)
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    model = Conformer(input_dim=392,
                      feat_dim=64,  # Reduced feature dimension
                      d_k=64,       # Reduced key dimension
                      d_v=64,       # Reduced value dimension
                      n_heads=8,    # Reduced number of attention heads
                      d_ff=2048,    # Reduced feed-forward dimension
                      max_len=100000,
                      dropout=0.1,
                      device=device,
                      n_lang=args.lang)  # Use 12 languages
    model.to(device)


    # --- Domain Adaptation: Load both source and target datasets ---
    source_txt = args.train  # Source domain labeled data
    target_txt = args.test   # Target domain (unlabeled for class, but used for domain)
    source_set = RawFeatures(source_txt)
    target_set = RawFeatures(target_txt)
    source_loader = DataLoader(dataset=source_set,
                               batch_size=args.batch,
                               pin_memory=True,
                               num_workers=4,
                               shuffle=True,
                               collate_fn=collate_fn_atten)
    target_loader = DataLoader(dataset=target_set,
                               batch_size=args.batch,
                               pin_memory=True,
                               num_workers=4,
                               shuffle=True,
                               collate_fn=collate_fn_atten)

    loss_func_CRE = nn.CrossEntropyLoss().to(device)
    loss_func_domain = nn.CrossEntropyLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_step = min(len(source_loader), len(target_loader))
    warm_up_with_cosine_lr = lambda step: step / args.warmup \
        if step <= args.warmup \
        else 0.5 * (math.cos((step - args.warmup) / (args.epochs * total_step - args.warmup) * math.pi) + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
    scaler = GradScaler(enabled=args.use_amp)

    logger = get_logger('exp6_da_sek_tytv.log')
    logger.info('start training:')

    # --- Training loop for domain adaptation ---
    best_balacc = 0.0
    best_epoch = -1
    for epoch in tqdm(range(args.epochs)):
        model.train()
        optimizer.zero_grad()
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        for step in range(total_step):
            # --- Get source batch (labeled) ---
            try:
                src_utt, src_labels, src_seq_len = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                src_utt, src_labels, src_seq_len = next(source_iter)
            src_utt = src_utt.to(device=device, dtype=torch.float)
            src_labels = src_labels.to(device=device, dtype=torch.long)
            src_mask = get_atten_mask(src_seq_len, src_utt.size(0)).to(device)

            # --- Get target batch (unlabeled for class, only for domain) ---
            try:
                tgt_utt, _, tgt_seq_len = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                tgt_utt, _, tgt_seq_len = next(target_iter)
            tgt_utt = tgt_utt.to(device=device, dtype=torch.float)
            tgt_mask = get_atten_mask(tgt_seq_len, tgt_utt.size(0)).to(device)

            # --- Domain labels: 0 for source, 1 for target ---
            src_domain_labels = torch.zeros(src_utt.size(0), dtype=torch.long, device=device)
            tgt_domain_labels = torch.ones(tgt_utt.size(0), dtype=torch.long, device=device)

            # --- Forward pass with GRL for both source and target ---
            with autocast(enabled=args.use_amp):
                # Source: class and domain outputs
                src_class_out, src_domain_out = model(src_utt, src_mask, grl_lambda=1.0, return_domain=True)
                # Target: only domain output
                _, tgt_domain_out = model(tgt_utt, tgt_mask, grl_lambda=1.0, return_domain=True)

                # --- Losses ---
                class_loss = loss_func_CRE(src_class_out, src_labels)
                domain_loss_src = loss_func_domain(src_domain_out, src_domain_labels)
                domain_loss_tgt = loss_func_domain(tgt_domain_out, tgt_domain_labels)
                domain_loss = (domain_loss_src + domain_loss_tgt) / 2.0
                total_loss = class_loss + domain_loss
                total_loss = total_loss / args.accum_steps

            # --- Backward pass ---
            scaler.scale(total_loss).backward()

            if (step + 1) % args.accum_steps == 0 or (step + 1) == total_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if step % 100 == 0:
                logger.info(f'Epoch {epoch+1}/{args.epochs} step: {step+1}/{total_step} class_loss: {class_loss.item():.4f} domain_loss: {domain_loss.item():.4f}')

            scheduler.step()

        # --- Validation and best model saving ---
        # Get balanced accuracy from validation (seen domain)
        cavg = validation(args.valid, model, args.model, device, kaldi=args.kaldi, num_lang=args.lang, msg = 'seen domain')
        # Parse balanced accuracy from log (or recompute here for robustness)
        # We'll recompute here for clarity
        model.eval()
        valid_set = RawFeatures(args.valid)
        valid_data = DataLoader(dataset=valid_set, batch_size=1, pin_memory=True, shuffle=False, collate_fn=collate_fn_atten)
        prediction_all = []
        labels_array = []
        with torch.no_grad():
            for step, (utt, labels, seq_len) in enumerate(valid_data):
                utt = utt.to(device=device, dtype=torch.float)
                labels = labels.to(device)
                atten_mask = get_atten_mask(seq_len, utt.size(0)).to(device=device)
                outputs = model(utt, atten_mask)
                predicted = torch.argmax(outputs, -1)
                prediction_all.append(predicted.cpu().numpy())
                labels_array.append(labels.cpu().numpy())
        import numpy as np
        prediction_all = np.concatenate(prediction_all)
        labels_array = np.concatenate(labels_array)
        balacc = balanced_accuracy_score(labels_array, prediction_all)
        if balacc > best_balacc:
            best_balacc = balacc
            best_epoch = epoch
            fname = f"bst_{epoch}_exp6_da_sek_tytv_{balacc:.4f}.ckpt"
            torch.save(model.state_dict(), fname)
            logger.info(f"Best model saved at epoch {best_epoch} with balanced accuracy {balacc:.4f}")

        # Also validate on unseen domain
        validation(args.test, model, args.model, device, kaldi=args.kaldi, num_lang=args.lang, msg = 'unseen domain')

        if epoch >= args.epochs - 5:
            torch.save(model.state_dict(), '{}_exp6_da_sek_tytv_{}.ckpt'.format(args.model, epoch))
            validation(args.valid, model, args.model, device, kaldi=args.kaldi, num_lang=args.lang, msg = 'seen domain')
            validation(args.test, model, args.model, device, kaldi=args.kaldi, num_lang=args.lang, msg = 'unseen domain')
    logging.info("Training completed")

if __name__ == "__main__":
    main()
