import torch


class L1:

    def __call__(self, batch_rep):
        return torch.sum(torch.abs(batch_rep), dim=-1).mean()


class L0:
    """non-differentiable
    """

    def __call__(self, batch_rep):
        return torch.count_nonzero(batch_rep, dim=-1).float().mean()


class FLOPS:
    """constraint from Minimizing FLOPs to Learn Efficient Sparse Representations
    https://arxiv.org/abs/2004.05665
    """

    def __call__(self, batch_rep):
        return torch.sum(torch.mean(torch.abs(batch_rep), dim=0) ** 2)


class RegWeightScheduler:
    """same scheduling as in: Minimizing FLOPs to Learn Efficient Sparse Representations
    https://arxiv.org/abs/2004.05665
    """

    def __init__(self, lambda_, T):
        self.lambda_ = lambda_
        self.T = T
        self.t = 0
        self.lambda_t = 0

    def step(self):
        """quadratic increase until time T
        """
        if self.t >= self.T:
            pass
        else:
            self.t += 1
            self.lambda_t = self.lambda_ * (self.t / self.T) ** 2
        return self.lambda_t

    def get_lambda(self):
        return self.lambda_t


class SparsityRatio:
    """non-differentiable
    """

    def __init__(self, output_dim):
        self.output_dim = output_dim

    def __call__(self, batch_rep):
        return 1 - torch.sum(batch_rep != 0, dim=-1).float().mean() / self.output_dim

import torch

import torch
from transformers import AutoTokenizer

class CasedRegularizer:
    """
    Penalizes L2 norm of activations at cased token indices.
    Equation:
        L_cased = w_t * sum_j || a_{j, I_c} ||_2^2
    """

    def __init__(self, tokenizer_name_or_obj, weight_scheduler=None, verbose=True):
        """
        Args:
            tokenizer_name_or_obj (str or PreTrainedTokenizer): HF tokenizer name or instance
            weight_scheduler (RegWeightScheduler or None): scheduler for w_t
            verbose (bool): if True, prints the cased tokens and their indices
        """
        if isinstance(tokenizer_name_or_obj, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_obj)
        else:
            tokenizer = tokenizer_name_or_obj
        
        vocab = tokenizer.get_vocab()  # {token: id}
        
        # Get all special tokens so we can exclude them
        special_tokens = set(tokenizer.all_special_tokens)
        
        # Filter tokens: must have uppercase, but not be a special token
        self.cased_tokens = [
            (tok, idx)
            for tok, idx in vocab.items()
            if any(ch.isupper() for ch in tok) and tok not in special_tokens
        ]
        self.cased_indices = torch.tensor([idx for _, idx in self.cased_tokens], dtype=torch.long)
        
        if verbose:
            print(f"[SPLADE CASED INDICES] Found {len(self.cased_tokens)} cased tokens.")
            preview = ", ".join([f"{tok}:{idx}" for tok, idx in self.cased_tokens[:50]])
            print(f"[SPLADE CASED INDICES] Example cased tokens: {preview}{' ...' if len(self.cased_tokens) > 50 else ''}")


    def __call__(self, batch_rep):
        """
        Args:
            batch_rep: [B, D] tensor of activations
        Returns:
            torch.Tensor: scalar loss
        """
        if len(self.cased_indices) == 0:
            return torch.tensor(0.0, device=batch_rep.device)

        # Select only the cased dimensions
        cased_activations = batch_rep.index_select(dim=-1, index=self.cased_indices.to(batch_rep.device))
        # L2 penalty per batch, then mean
        penalty = torch.sum(cased_activations ** 2, dim=-1).mean()

        # Apply schedule if provided
        if self.weight_scheduler is not None:
            w_t = self.weight_scheduler.step()
            return w_t * penalty
        else:
            return penalty


def init_regularizer(reg, **kwargs):
    if reg == "L0":
        return L0()
    elif reg == "sparsity_ratio":
        return SparsityRatio(output_dim=kwargs["output_dim"])
    elif reg == "L1":
        return L1()
    elif reg == "FLOPS":
        return FLOPS()
    elif reg == "CASED":
        # Expect "cased_indices" in kwargs
        return CasedRegularizer(
            tokenizer_name_or_obj=kwargs["tokenizer_name_or_obj"],
            weight_scheduler=kwargs.get("weight_scheduler", None)
        )
    else:
        raise NotImplementedError("provide valid regularizer")

