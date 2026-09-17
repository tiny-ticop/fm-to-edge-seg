from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from PIL import Image, ImageEnhance, ImageOps


@dataclass
class BinarySegmentationAugmentation:
    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.5
    rotate_90_probability: float = 0.5
    brightness: float = 0.1
    contrast: float = 0.1

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image,
        dense_targets: list[Image.Image] | None = None,
        feature_target: torch.Tensor | None = None,
    ) -> tuple:
        targets = dense_targets or []
        if random.random() < self.horizontal_flip_probability:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)
            targets = [ImageOps.mirror(target) for target in targets]
            if feature_target is not None:
                feature_target = torch.flip(feature_target, dims=(-1,))
        if random.random() < self.vertical_flip_probability:
            image = ImageOps.flip(image)
            mask = ImageOps.flip(mask)
            targets = [ImageOps.flip(target) for target in targets]
            if feature_target is not None:
                feature_target = torch.flip(feature_target, dims=(-2,))
        if random.random() < self.rotate_90_probability:
            rotations = random.randint(1, 3)
            image = image.rotate(90 * rotations, expand=True)
            mask = mask.rotate(90 * rotations, resample=Image.Resampling.NEAREST, expand=True)
            targets = [
                target.rotate(90 * rotations, resample=Image.Resampling.BILINEAR, expand=True)
                for target in targets
            ]
            if feature_target is not None:
                feature_target = torch.rot90(feature_target, rotations, dims=(-2, -1))
        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            image = ImageEnhance.Brightness(image).enhance(factor)
        if self.contrast > 0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            image = ImageEnhance.Contrast(image).enhance(factor)
        if dense_targets is None and feature_target is None:
            return image, mask
        if feature_target is not None:
            return image, mask, targets, feature_target
        return image, mask, targets
