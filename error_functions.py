import torch
import numpy as np
import pandas as pd
import os
import sys

def kabsch_align(P, Q):
    P_cent = P - P.mean(dim=0)
    Q_cent = Q - Q.mean(dim=0)
    C = torch.matmul(P_cent.T, Q_cent)
    V, S, W = torch.linalg.svd(C)
    d = torch.det(torch.matmul(W.T, V.T))
    D = torch.diag(torch.tensor([1.0, 1.0, d]))
    U = torch.matmul(W.T, torch.matmul(D, V.T))
    return torch.matmul(P_cent, U)

def rmsd_loss(pred_coords, true_coords, batch):
    loss = 0.0
    for i in batch.unique():
        idx = (batch == i)
        aligned = kabsch_align(pred_coords[idx], true_coords[idx])
        loss += torch.mean((aligned - true_coords[idx]) ** 2)
    return torch.sqrt(loss / len(batch.unique()))

def orientation_invariant_rmsd(pred, true, mask, eps=1e-8):
    batch_size = pred.shape[0]
    rmsd_list = []

    for b in range(batch_size):
        m = mask[b].bool()
        X = pred[b][m]
        Y = true[b][m]

        if X.shape[0] < 3:
            continue

        # Center
        X_mean = X.mean(dim=0)
        Y_mean = Y.mean(dim=0)
        X_centered = X - X_mean
        Y_centered = Y - Y_mean

        # Detach and convert to NumPy for SVD
        # C = (X_centered.T @ Y_centered).cpu().numpy()
        C = (X_centered.T @ Y_centered).detach().cpu().numpy()

        try:
            V, S, Wt = svd(C)
        except Exception as e:
            print(f"Skipping sample {b} due to SVD failure:", e)
            continue

        d = np.linalg.det(V @ Wt)
        D = np.diag([1.0, 1.0, d])
        U = torch.tensor(V @ D @ Wt, device=pred.device, dtype=pred.dtype)

        # Align X in PyTorch
        X_aligned = X_centered @ U

        diff = X_aligned - Y_centered
        rmsd = torch.sqrt((diff ** 2).sum() / X.shape[0] + eps)
        rmsd_list.append(rmsd)

    if len(rmsd_list) == 0:
        return (pred.sum() * 0.0) + 0.0  # maintain autograd graph

    return torch.stack(rmsd_list).mean()

def tm_score(pred_coords: torch.Tensor,
             true_coords: torch.Tensor,
             mask: torch.Tensor = None) -> torch.Tensor:
    """
    Computes TM-score between predicted and true coordinates.
    
    Args:
        pred_coords: (L, 3) predicted structure (should be aligned to reference)
        true_coords: (L, 3) reference structure
        mask: (L,) optional binary mask of valid positions
    
    Returns:
        TM-score (scalar tensor)
    """
    assert pred_coords.shape == true_coords.shape
    L = pred_coords.shape[0]

    if mask is None:
        mask = torch.ones(L, dtype=torch.float32, device=pred_coords.device)
    
    # Select valid residues
    valid = mask > 0
    pred = pred_coords[valid]
    true = true_coords[valid]
    L_valid = valid.sum().item()

    if L_valid == 0:
        return torch.tensor(0.0, device=pred_coords.device)

    # Compute distances
    dist = torch.norm(pred - true, dim=-1)  # shape: (L_valid,)

    # Avoid complex values if L_valid < 15
    L_clamped = max(L_valid, 19)  # to keep d0 real-valued and non-zero
    d0 = 1.24 * (L_clamped - 15)**(1/3) - 1.8
    d0 = max(d0, 0.5)

    score = 1.0 / (1.0 + (dist / d0) ** 2)
    return score.mean()

def soft_tm_loss(pred, true, mask, d0=1.24, eps=1e-8):
    """
    Differentiable TM-like loss: 1 - soft_TM_score
    """
    mask = mask.unsqueeze(-1)  # [B, L, 1]
    pred = pred * mask
    true = true * mask

    # Center
    pred_centered = pred - (pred.sum(dim=1, keepdim=True) / (mask.sum(dim=1, keepdim=True) + eps))
    true_centered = true - (true.sum(dim=1, keepdim=True) / (mask.sum(dim=1, keepdim=True) + eps))

    # Pairwise distances
    pred_dist = torch.cdist(pred_centered, pred_centered, p=2)
    true_dist = torch.cdist(true_centered, true_centered, p=2)

    # Avoid zero-distance pairs
    valid = (mask @ mask.transpose(1, 2)) > 0
    score = 1 / (1 + ((pred_dist - true_dist) / d0) ** 2)
    tm_loss = (1 - score)[valid].mean()

    return tm_loss


def lddt_torch(predicted_points: torch.Tensor,
               true_points: torch.Tensor,
               true_mask: torch.Tensor,
               cutoff: float = 15.0,
               per_residue: bool = False) -> torch.Tensor:
    """
    Computes approximate lDDT score between predicted and true 3D coordinates.

    Args:
        predicted_points: (L, 3) predicted 3D coordinates
        true_points: (L, 3) ground-truth 3D coordinates
        true_mask: (L,) binary tensor indicating valid positions
        cutoff: float, max distance in true structure to consider pair
        per_residue: bool, whether to return per-residue scores

    Returns:
        lDDT score (scalar) or (L,) tensor if per_residue=True
    """
    # Ensure proper shapes
    assert predicted_points.shape == true_points.shape
    assert predicted_points.shape[-1] == 3
    L = predicted_points.shape[0]

    # Mask reshaped for broadcasting
    true_mask = true_mask.float()
    mask_2d = true_mask[:, None] * true_mask[None, :]
    non_self = 1.0 - torch.eye(L, device=true_points.device)

    # Pairwise distances
    dmat_true = torch.cdist(true_points, true_points, p=2)
    dmat_pred = torch.cdist(predicted_points, predicted_points, p=2)

    # Mask for distances within cutoff in the *true* structure
    pair_mask = ((dmat_true < cutoff).float() *
                 mask_2d * non_self)

    # Absolute distance differences
    dist_l1 = torch.abs(dmat_true - dmat_pred)

    # Scoring: 4 thresholds
    score = 0.25 * ((dist_l1 < 0.5).float() +
                    (dist_l1 < 1.0).float() +
                    (dist_l1 < 2.0).float() +
                    (dist_l1 < 4.0).float())

    masked_score = pair_mask * score

    # Normalize
    if per_residue:
        norm = (pair_mask.sum(dim=1) + 1e-10)  # (L,)
        per_res_score = masked_score.sum(dim=1) / norm
        return per_res_score
    else:
        total_score = masked_score.sum()
        norm = pair_mask.sum() + 1e-10
        return total_score / norm

def composite_loss(pred, true, mask, w_rmsd=1.0, w_tm=1.0, w_lddt=0.2):
    rmsd = orientation_invariant_rmsd(pred, true, mask)
    tm = soft_tm_loss(pred, true, mask)
    lddt = relative_geometry_loss(pred, true, mask)
    return w_rmsd * rmsd + w_tm * tm + w_lddt * lddt