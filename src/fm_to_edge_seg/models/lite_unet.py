from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class StudentFeatures(NamedTuple):
    encoder_s4: torch.Tensor
    encoder_s8: torch.Tensor
    encoder_s16: torch.Tensor
    encoder_s32: torch.Tensor
    decoder_s4: torch.Tensor


class DepthwiseSeparableBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(
                input_channels,
                input_channels,
                kernel_size=3,
                padding=1,
                groups=input_channels,
                bias=False,
            ),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )


class MobileNetV3LiteUNet(nn.Module):
    """MobileNetV3-Small encoder with a quantization-friendly lightweight decoder."""

    feature_channels = {
        "encoder_s4": 16,
        "encoder_s8": 24,
        "encoder_s16": 48,
        "encoder_s32": 576,
        "decoder_s4": 32,
    }

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.encoder = mobilenet_v3_small(weights=weights).features

        self.deep_projection = nn.Sequential(
            nn.Conv2d(576, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )
        self.decode_s16 = DepthwiseSeparableBlock(96 + 48, 64)
        self.decode_s8 = DepthwiseSeparableBlock(64 + 24, 48)
        self.decode_s4 = DepthwiseSeparableBlock(48 + 16, 32)
        self.decode_s2 = DepthwiseSeparableBlock(32 + 16, 24)
        self.classifier = nn.Conv2d(24, 1, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_features(image)
        return logits

    def forward_with_features(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, StudentFeatures]:
        input_size = image.shape[-2:]
        encoder_features = self._extract_encoder_features(image)

        decoded = self.deep_projection(encoder_features["s32"])
        decoded = self._upsample_and_decode(decoded, encoder_features["s16"], self.decode_s16)
        decoded = self._upsample_and_decode(decoded, encoder_features["s8"], self.decode_s8)
        decoded_s4 = self._upsample_and_decode(
            decoded,
            encoder_features["s4"],
            self.decode_s4,
        )
        decoded = self._upsample_and_decode(decoded_s4, encoder_features["s2"], self.decode_s2)
        logits = self.classifier(decoded)
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)

        features = StudentFeatures(
            encoder_s4=encoder_features["s4"],
            encoder_s8=encoder_features["s8"],
            encoder_s16=encoder_features["s16"],
            encoder_s32=encoder_features["s32"],
            decoder_s4=decoded_s4,
        )
        return logits, features

    def _extract_encoder_features(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        feature = image
        for index, layer in enumerate(self.encoder):
            feature = layer(feature)
            if index == 0:
                outputs["s2"] = feature
            elif index == 1:
                outputs["s4"] = feature
            elif index == 3:
                outputs["s8"] = feature
            elif index == 8:
                outputs["s16"] = feature
            elif index == 12:
                outputs["s32"] = feature
        return outputs

    @staticmethod
    def _upsample_and_decode(
        feature: torch.Tensor,
        skip: torch.Tensor,
        block: nn.Module,
    ) -> torch.Tensor:
        feature = F.interpolate(
            feature,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return block(torch.cat((feature, skip), dim=1))
