from __future__ import annotations

import random
from dataclasses import dataclass

from PIL import Image, ImageEnhance, ImageOps


@dataclass
class BinarySegmentationAugmentation:
    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.5
    rotate_90_probability: float = 0.5
    brightness: float = 0.1
    contrast: float = 0.1

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if random.random() < self.horizontal_flip_probability:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)
        if random.random() < self.vertical_flip_probability:
            image = ImageOps.flip(image)
            mask = ImageOps.flip(mask)
        if random.random() < self.rotate_90_probability:
            rotations = random.randint(1, 3)
            image = image.rotate(90 * rotations, expand=True)
            mask = mask.rotate(90 * rotations, resample=Image.Resampling.NEAREST, expand=True)
        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            image = ImageEnhance.Brightness(image).enhance(factor)
        if self.contrast > 0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            image = ImageEnhance.Contrast(image).enhance(factor)
        return image, mask
