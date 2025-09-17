import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_util
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from transformer import *
import conformer as cf
from convolution_module import Conv1dSubampling

# --- GRL and Domain Classifier for Domain Adaptation ---
from torch.autograd import Function

# Gradient Reversal Layer
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class Conformer(nn.Module):
    def __init__(self, input_dim, feat_dim, d_k, d_v, n_heads, d_ff, max_len, dropout, device, n_lang):
        super(Conformer, self).__init__()
        self.device = device

        # Reduced feature dimensions and number of attention heads
        self.conv_subsample = Conv1dSubampling(in_channels=input_dim, out_channels=input_dim)
        self.transform = nn.Linear(input_dim, feat_dim)  # Reduced feat_dim
        self.dropout = nn.Dropout(dropout)
        self.d_model = feat_dim * n_heads  # Reduced d_model
        self.layernorm1 = LayerNorm(feat_dim)
        self.n_heads = n_heads
        self.attention_block1 = cf.ConformerEncoder(self.d_model, d_k, d_v, d_ff, n_heads, dropout, max_len, device)
        self.attention_block2 = cf.ConformerEncoder(self.d_model, d_k, d_v, d_ff, n_heads, dropout, max_len, device)
        self.fc1 = nn.Linear(self.d_model * 2, self.d_model)  # Reduced fully connected layers
        self.fc2 = nn.Linear(self.d_model, self.d_model)
        self.fc3 = nn.Linear(self.d_model, n_lang)

        # --- Domain adaptation components ---
        self.grl = GradientReversalLayer(lambda_=1.0)  # lambda_ can be scheduled during training
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 2)  # 2 domains: source=0, target=1
        )

    def forward(self, x, atten_mask, grl_lambda=1.0, return_domain=False):
        batch_size = x.size(0)
        output = self.transform(x)  # x [B, T, 768] => [B, T, feat_dim]
        output = self.layernorm1(output)
        output = output.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output, _ = self.attention_block1(output, atten_mask)
        output, _ = self.attention_block2(output, atten_mask)
        stats = torch.cat((output.mean(dim=1), output.std(dim=1)), dim=1)  # [B, d_model*2]

        # --- Main task head ---
        class_out = F.relu(self.fc1(stats))
        class_out = F.relu(self.fc2(class_out))
        class_out = self.fc3(class_out)

        # --- Domain classifier head with GRL ---
        grl_stats = self.grl(stats)
        self.grl.lambda_ = grl_lambda  # allow dynamic lambda
        domain_out = self.domain_classifier(grl_stats)

        if return_domain:
            return class_out, domain_out
        else:
            return class_out

