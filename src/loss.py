from typing import Optional

import torch
import torch.nn.functional as F
import torch.nn as nn


class CLIPContrastiveLoss(nn.Module):
    def __init__(
        self,
        local_loss=False,
        cache_labels=False,
        rank=0,
        world_size=1,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size

        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def forward(self, logits_per_image, logits_per_text, output_dict=False):
        device = logits_per_image.device

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss



class SigLIPLoss(nn.Module):
    def __init__(
        self,
        cache_labels=False,
        rank=0,
        world_size=1,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size

        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        """
        Generates a target label matrix Y in {-1, 1}^(B x B).
        Diagonal entries are 1 (positive pairs), off-diagonals are -1 (negative pairs).
        """
        if self.prev_num_logits != num_logits or device not in self.labels:
            # Construct square target matrix: -1 for negatives, +1 for positives on diagonal
            labels = -torch.ones((num_logits, num_logits), device=device)
            
            # In distributed training with local loss, shift diagonal offset by rank
            diag_idx = torch.arange(num_logits, device=device)
            if self.world_size > 1:
                # Offset row index relative to column block rank
                labels[diag_idx, diag_idx] = 1.0
            else:
                labels.fill_diagonal_(1.0)

            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        return labels

    def forward(self, logits_per_image, logits_per_text=None, output_dict=False):
        """
        Args:
            logits_per_image: Pre-computed logits matrix (image_embeds @ text_embeds.T) * logit_scale + logit_bias.
                              Shape: (batch_size, batch_size)
            logits_per_text: Unused for SigLIP (retained for signature parity with CLIP drop-in usage),
                             since image-to-text and text-to-image pair matrices are symmetric transpose.
        """
        device = logits_per_image.device
        num_logits = logits_per_image.shape[0]

        # Targets: +1 on diagonal, -1 off-diagonal
        labels = self.get_ground_truth(device, num_logits)

        # SigLIP Loss formulation: - 1/N * sum( log_sigmoid( y_ij * z_ij ) )
        # F.logsigmoid(z * y) is mathematically identical to -BCEWithLogits(z, (y + 1) / 2)
        loss = -F.logsigmoid(logits_per_image * labels).sum() / num_logits

        return {"contrastive_loss": loss} if output_dict else loss


# =============================================================================
# Generalised losses used by the multi-loss training (global / dense / region).
#
# Both take `logits` (R, C) and a boolean `pos` (R, C) marking which (row, col) pairs are
# positives. A row/column may have several positives (a dense caption has 3-6 sentences,
# the same keyword text is shared by many regions) or none (hard-negative sentences are
# columns without any positive). With pos = eye(B) they reduce exactly to
# CLIPContrastiveLoss / SigLIPLoss above.
# =============================================================================


def _mean_over_valid(x: torch.Tensor, valid: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if valid.any():
        return x[valid].mean()
    return ref.sum() * 0.0  # keeps the graph alive for DDP when there is nothing to learn from


class MultiPositiveCLIPLoss(nn.Module):
    """Symmetric softmax contrastive loss with any number of positives per row / column.

    image->text: for every row with >=1 positive, -mean_{p in pos(row)} log softmax_row[p].
    text->image: same over columns. Rows/columns without a positive are skipped in that
    direction but still act as negatives in the other one (that is how hard negatives work).
    """

    def forward(self, logits: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        posf = pos.to(logits.dtype)

        n_row = posf.sum(1)
        row = -(F.log_softmax(logits, dim=1) * posf).sum(1) / n_row.clamp_min(1.0)
        n_col = posf.sum(0)
        col = -(F.log_softmax(logits, dim=0) * posf).sum(0) / n_col.clamp_min(1.0)

        return 0.5 * (_mean_over_valid(row, n_row > 0, logits) + _mean_over_valid(col, n_col > 0, logits))


class MultiPositiveSigLIPLoss(nn.Module):
    """Pairwise sigmoid loss; label +1 for positives, -1 for everything else.

    Normalised by the number of positive pairs (== B for a B x B diagonal, i.e. the
    original SigLIP normalisation), so losses with many more negatives per positive
    (dense, region) stay on the same scale as the global loss.
    """

    def forward(self, logits: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        y = pos.to(logits.dtype) * 2.0 - 1.0
        return -F.logsigmoid(y * logits).sum() / pos.sum().clamp_min(1)


def build_pair_loss(loss_type: str) -> nn.Module:
    if loss_type == "clip":
        return MultiPositiveCLIPLoss()
    if loss_type == "siglip":
        return MultiPositiveSigLIPLoss()
    raise ValueError(f"loss_type must be 'clip' or 'siglip', got {loss_type!r}")


class NegatedKeywordLoss(nn.Module):
    """Push images away from false statements such as "there is no solar panel".

    img_emb (B, D) and txt_emb (U, D) are L2-normalised; (kw_img[n], kw_txt[n]) lists the
    image / negated-sentence pairs. Returns the MEAN cosine similarity over the pairs (the
    sum divided by the number of pairs, so its scale does not depend on how many keywords
    were sampled); minimising it lowers the similarity. With `margin` set it becomes
    mean(relu(cos - margin)), which stops pushing once a pair is already dissimilar enough.
    """

    def __init__(self, margin: Optional[float] = None):
        super().__init__()
        self.margin = margin

    def forward(self, img_emb, txt_emb, kw_img, kw_txt):
        sims = (img_emb[kw_img] * txt_emb[kw_txt]).sum(-1)
        if self.margin is not None:
            sims = F.relu(sims - self.margin)
        return sims.mean()
