import torch
from .main import CLIPLightningModule


def get_clip(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    lightning_model = CLIPLightningModule(**ckpt["hyper_parameters"]).to(device)
    lightning_model.load_state_dict(ckpt["state_dict"])
    lightning_model.eval()
    return lightning_model.model
