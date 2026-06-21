import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def generate_masks(N, s, p1, input_size, device='cpu'):
    cell_size = np.ceil(np.array(input_size) / s).astype(int)
    up_size = (int((s + 1) * cell_size[0]), int((s + 1) * cell_size[1]))

    grid = (torch.rand(N, 1, s, s, device=device) < p1).float()

    masks = torch.empty(N, *input_size, device=device)

    for i in tqdm(range(N), desc='Generating masks'):
        # Random shifts
        x = np.random.randint(0, cell_size[0])
        y = np.random.randint(0, cell_size[1])
        # Linear upsampling and cropping
        upsampled = F.interpolate(grid[i:i + 1], size=up_size, mode='bilinear', align_corners=False)
        masks[i] = upsampled[0, 0, x:x + input_size[0], y:y + input_size[1]]

    masks = masks.unsqueeze(1)  # (N, 1, H, W), broadcasts over channels
    return masks


def explain(model, inp, masks, p1, batch_size=100, device='cpu'):
    N = masks.shape[0]
    preds = []
    # inp: (1, C, H, W), masks: (N, 1, H, W) -> masked: (N, C, H, W)
    masked = inp * masks
    with torch.no_grad():
        for i in tqdm(range(0, N, batch_size), desc='Explaining'):
            batch = masked[i:min(i + batch_size, N)].to(device)
            preds.append(model(batch).reshape(batch.shape[0], -1))
    preds = torch.cat(preds, dim=0)  # (N, num_classes)

    sal = (preds.to(torch.float32)).t().mm((masks.to(torch.float32)).view(N, -1)).view(-1, *inp.shape[-2:])
    sal = sal / N / p1
    return sal
