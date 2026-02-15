import random
from typing import Iterable, Sequence

import torch

class CutMix:
    """Simple CutMix for a sequence of tensor images (C,H,W).

    images: list/tuple of torch.Tensor with identical spatial size.
    alpha: Beta distribution parameter; alpha<=0 disables mixing.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def _rand_bbox(self, width: int, height: int, lam: float):
        cut_w = int(width * (1 - lam) ** 0.5)
        cut_h = int(height * (1 - lam) ** 0.5)
        cx = random.randint(0, width)
        cy = random.randint(0, height)
        x1 = max(cx - cut_w // 2, 0)
        y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, width)
        y2 = min(cy + cut_h // 2, height)
        return x1, y1, x2, y2

    def __call__(self, images: Iterable[torch.Tensor]) -> tuple:
        images = tuple(images)
        if len(images) < 2 or self.alpha <= 0:
            return images

        lam = torch.distributions.beta.Beta(self.alpha, self.alpha).sample().item()
        c, h, w = images[0].shape
        x1, y1, x2, y2 = self._rand_bbox(w, h, lam)
        src_idx = random.randrange(len(images))

        mixed = []
        for img in images:
            patch = images[src_idx][..., y1:y2, x1:x2]
            out = img.clone()
            out[..., y1:y2, x1:x2] = patch
            mixed.append(out)
        return tuple(mixed)
