import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', "1,2")

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


# =============================================================================
# Counterfactual margin / threshold / uniformity / silhouette.
#
# All embeddings passed in are already L2-normalised, so the dot product is the cosine
# similarity s(.,.). `pos` / `neg` are boolean (B, U) masks over a shared pool of U texts
# (a given text may be a positive for image a and a negative for image b).
# =============================================================================


def _zero(ref: torch.Tensor) -> torch.Tensor:
    return ref.sum() * 0.0  # keeps the graph (and DDP) alive when a group is empty


class CounterfactualMarginLoss(nn.Module):
    r"""``mean Softplus[ (s(I, a-) - s(I, a+) + m * (1 - s(a+, a-))) / tau ]`` over all
    positive x negative pairs of every image.

    A negative that is textually close to a positive (``s(a+, a-)`` near 1) gets its margin
    relaxed, so the model only has to separate them as much as their wording differs. Works for
    both the dense negatives and the keyword negatives.
    """

    def __init__(self, margin: float = 0.2, tau: float = 0.1, tau_sim: float = 0.1):
        super().__init__()
        self.margin = margin
        self.tau = max(tau, 1e-6)
        # scale the text-text similarity term into [0, 1] so m does not depend on tau
        self.tau_sim = max(tau_sim, 1e-6)

    def forward(self, img_emb: torch.Tensor, txt_emb: torch.Tensor,
                pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        B = img_emb.shape[0]
        if txt_emb.shape[0] == 0:
            return _zero(img_emb)
        sim_im = img_emb @ txt_emb.t()          # (B, U)
        sim_tt = txt_emb @ txt_emb.t()          # (U, U)
        terms = []
        for i in range(B):
            p = pos[i].nonzero(as_tuple=True)[0]
            n = neg[i].nonzero(as_tuple=True)[0]
            if p.numel() == 0 or n.numel() == 0:
                continue
            s_pos = sim_im[i, p]                 # (P,)
            s_neg = sim_im[i, n]                 # (N,)
            cross = sim_tt[p][:, n]              # (P, N) cosine between each a+ and a-
            margin = self.margin * (1.0 - cross)    # 16-10 I am removing the tau_sim as it causes inconsistency
            terms.append(F.softplus((s_neg.unsqueeze(0) - s_pos.unsqueeze(1) + margin) / self.tau))
        if not terms:
            return _zero(img_emb)
        return torch.cat([t.flatten() for t in terms]).mean()


class ThresholdLoss(nn.Module):
    r"""``mean_{pos} [t+ - s]+^2 + mean_{neg} [s - t-]+^2`` with ``[x]+ = max(0, x)``.

    Pulls every positive similarity above ``t_pos`` and every negative similarity below ``t_neg``
    (a margin-style hinge, but squared). Positives / negatives come from the dense and keyword
    groups; the two sides are normalised independently.
    """

    def __init__(self, t_pos: float = 0.5, t_neg: float = 0.1):
        super().__init__()
        self.t_pos = t_pos
        self.t_neg = t_neg

    def forward(self, img_emb: torch.Tensor, txt_emb: torch.Tensor,
                pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        if txt_emb.shape[0] == 0:
            return _zero(img_emb)
        sim = img_emb @ txt_emb.t()          # (B, U)
        loss = _zero(img_emb)
        pos_v = sim[pos]
        if pos_v.numel():
            loss = loss + F.relu(self.t_pos - pos_v).pow(2).mean()
        neg_v = sim[neg]
        if neg_v.numel():
            loss = loss + F.relu(neg_v - self.t_neg).pow(2).mean()
        return loss


class UniformityLoss(nn.Module):
    r"""``log E_{i != j}[ exp(-2 ||z_i - z_j||^2) ]`` (Wang & Isola).

    Lower is more uniform; computed stably as ``logsumexp(-2 d^2) - log(n(n-1))``.
    """

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        n = z.shape[0]
        if n < 2:
            return _zero(z)
        d2 = torch.cdist(z, z).pow(2)
        eye = torch.eye(n, dtype=torch.bool, device=z.device)
        vals = (-2.0 * d2)[~eye]
        return torch.logsumexp(vals, dim=0) - torch.log(torch.tensor(vals.numel(), dtype=z.dtype, device=z.device))


class SilhouetteLoss(nn.Module):
    r"""``-mean silhouette`` over embeddings ``z`` (N, D) with integer ``labels`` (N,).

    a(i) is the mean distance to the other members of i's cluster, b(i) the smallest mean
    distance to a different cluster, silhouette = (b - a) / max(a, b). Minimising the loss
    maximises the silhouette (tight, well-separated clusters). Singleton clusters are handled
    (a(i) = 0). Vectorised over points and clusters.
    """

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        n = z.shape[0]
        if n < 3 or labels.max().item() < 1:
            return _zero(z)
        labels = labels.to(torch.long)
        C = int(labels.max().item()) + 1
        D = torch.cdist(z, z)
        M = F.one_hot(labels, C).to(z.dtype)         # (N, C) one-hot
        counts = M.sum(0)                            # (C,)
        sums = D @ M                                 # (N, C) sum of distances to each cluster
        idx = torch.arange(n, device=z.device)
        a = sums[idx, labels] / (counts[labels] - 1).clamp_min(1)   # self distance is 0
        means = sums / counts.clamp_min(1)
        means[idx, labels] = float("inf")            # ignore the point's own cluster for b
        b = means.min(1).values
        sil = (b - a) / torch.maximum(a, b).clamp_min(1e-8)
        return -sil.mean()


class RelationalLoss(nn.Module):
    r"""Keep the *pairwise relational structure* of the pre-projector features after projection.

    ``pre`` are the frozen backbone features (the precomputed image CLS / text EOS states coming
    out of the ``.npy`` files) and ``post`` are the same samples after the linear projector. For
    every selected pair ``(i, j)`` the loss asks for the cosine similarities to agree::

        loss = (1 / B) * sum_i [ cos(pre_i, pre_j) - cos(post_i, post_j) ]

    and the caller adds the image and text terms. ``pre`` is detached: only the projected space is
    shaped. ``pairing`` chooses ``j``:

    * ``"perm"``  -- a random bijection over the batch (default; exactly ``1/B`` pairs);
    * ``"shift"`` -- ``j = (i + 1) % B`` (deterministic);
    * ``"all"``   -- every off-diagonal pair (the mean is over ``B * (B - 1)`` pairs).

    Note: minimising the *raw* difference pushes every ``cos(post_i, post_j)`` towards +1 (the
    representation-collapse direction). ``squared=True`` turns it into the usual structure-matching
    objective ``mean((cos_pre - cos_post)^2)``, which is minimised when the two structures agree.
    """

    def __init__(self, pairing: str = "perm", squared: bool = False):
        super().__init__()
        if pairing not in ("perm", "shift", "all"):
            raise ValueError(f"pairing must be 'perm', 'shift' or 'all', got {pairing!r}")
        self.pairing = pairing
        self.squared = squared

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        pre = F.normalize(pre.detach().to(post.dtype), dim=-1)
        post = F.normalize(post, dim=-1)
        b = pre.shape[0]
        if b < 2:
            return _zero(post)

        if self.pairing == "all":
            sim_pre = pre @ pre.t()
            sim_post = post @ post.t()
            off = ~torch.eye(b, dtype=torch.bool, device=pre.device)
            diff = (sim_pre - sim_post)[off]
        else:
            idx = ((torch.arange(b, device=pre.device) + 1) % b if self.pairing == "shift"
                   else torch.randperm(b, device=pre.device))
            diff = (pre * pre[idx]).sum(-1) - (post * post[idx]).sum(-1)

        return (diff * diff).mean() if self.squared else diff.mean()
