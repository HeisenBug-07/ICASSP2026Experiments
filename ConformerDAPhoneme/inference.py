import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
import logging
import scoring
import compute_eer
from data_load import RawFeatures, collate_fn_atten
from model import Conformer  # Import the Conformer model

def evaluate_conformer(model_path, test_txt, num_lang, device, kaldi_root=''):
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if the checkpoint contains 'state_dict'
    if 'state_dict' in checkpoint:
        model_weights = checkpoint['state_dict']
        model_params = checkpoint.get('model_params', {})  # Retrieve model parameters if available
    else:
        model_weights = checkpoint  # Assume the checkpoint contains only model weights
        model_params = {}

    # Load dataset
    test_set = RawFeatures(test_txt)
    test_loader = DataLoader(dataset=test_set,
                            batch_size=1,
                            pin_memory=True,
                            shuffle=False,
                            collate_fn=collate_fn_atten)

    # Load pretrained model
    model = Conformer(
        input_dim=model_params.get('input_dim', 39),  # Default to 39 if not specified
        feat_dim=model_params.get('feat_dim', 64),
        d_k=model_params.get('d_k', 64),
        d_v=model_params.get('d_v', 64),
        n_heads=model_params.get('n_heads', 8),
        d_ff=model_params.get('d_ff', 2048),
        max_len=model_params.get('max_len', 100000),
        dropout=model_params.get('dropout', 0.1),
        device=device,
        n_lang=num_lang
    )
    model.load_state_dict(model_weights)  # Load model weights
    model.to(device)
    model.eval()

    # Evaluation containers
    all_scores = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for utt, labels, seq_len in tqdm(test_loader, desc="Evaluating"):
            utt = utt.to(device, dtype=torch.float)
            labels = labels.to(device)
            atten_mask = get_atten_mask(seq_len, utt.size(0)).to(device)
            
            outputs = model(utt, atten_mask)
            scores = torch.softmax(outputs, dim=-1)
            
            all_scores.append(scores.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_preds.append(torch.argmax(outputs, -1).cpu().numpy())

    # Process results
    scores = np.concatenate(all_scores)
    labels_array = np.concatenate(all_labels)
    preds_array = np.concatenate(all_preds)

    # Calculate metrics
    accuracy = (preds_array == labels_array).mean()
    balanced_acc = balanced_accuracy_score(labels_array, preds_array)
    
    # Prepare trial and score files
    trial_txt = 'conformer_eval_trial.txt'
    score_txt = 'conformer_eval_scores.txt'
    
    scoring.get_trials(test_txt, num_lang, trial_txt)
    scoring.get_score(test_txt, scores, num_lang, score_txt)
    
    # Compute EER and Cavg
    target_score, non_target_score = compute_eer.load_file(score_txt, trial_txt)
    p_miss, p_fa = compute_eer.compute_rocch(target_score, non_target_score)
    eer = compute_eer.rocch2eer(p_miss, p_fa)
    cavg = scoring.compute_cavg(trial_txt, score_txt)

    # Cleanup temporary files
    os.remove(trial_txt)
    os.remove(score_txt)

    print("\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"EER: {eer:.4f}")
    print(f"Cavg: {cavg:.4f}")

    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'eer': eer,
        'cavg': cavg
    }

# Usage example
if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    results = evaluate_conformer(
        model_path="conformer_ekstep_12_mfcc_epoch4.ckpt",
        test_txt="ekstep_12_test_mfcc.txt",
        num_lang=12,
        device=device,
        kaldi_root="/path/to/kaldi"
    )