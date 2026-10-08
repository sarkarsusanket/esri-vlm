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