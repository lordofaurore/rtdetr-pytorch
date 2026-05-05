"""
by lyuwenyu
"""

import torch 
import torch.nn as nn 

import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision import tv_tensors as datapoints

import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from torchvision.transforms.v2 import ToTensor, ConvertImageDtype, SanitizeBoundingBoxes

from PIL import Image 
from typing import Any, Dict, List, Optional

from src.core import register, GLOBAL_CONFIG


__all__ = ['Compose', ]


# Compatibility wrappers for torchvision 0.22.0
class ToImageTensor:
    def __init__(self, **kwargs):
        self.transform = ToTensor(**kwargs)
    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)
    def __repr__(self):
        return repr(self.transform)

class ConvertDtype:
    def __init__(self, **kwargs):
        self.transform = ConvertImageDtype(**kwargs)
    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)
    def __repr__(self):
        return repr(self.transform)

class SanitizeBoundingBox:
    def __init__(self, **kwargs):
        self.transform = SanitizeBoundingBoxes(**kwargs)
    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)
    def __repr__(self):
        return repr(self.transform)


# Register standard transforms
RandomPhotometricDistort = register(T.RandomPhotometricDistort)
RandomZoomOut = register(T.RandomZoomOut)
RandomHorizontalFlip = register(T.RandomHorizontalFlip)
Resize = register(T.Resize)
Normalize = register(T.Normalize)
RandomCrop = register(T.RandomCrop)

# Register compatibility wrappers
ToImageTensor = register(ToImageTensor)
ConvertDtype = register(ConvertDtype)
SanitizeBoundingBox = register(SanitizeBoundingBox)


@register
class Compose(T.Compose):
    def __init__(self, ops) -> None:
        transforms = []
        if ops is not None:
            for op in ops:
                if isinstance(op, dict):
                    name = op.pop('type')
                    transfom = getattr(GLOBAL_CONFIG[name]['_pymodule'], name)(**op)
                    transforms.append(transfom)
                elif isinstance(op, nn.Module):
                    transforms.append(op)
                else:
                    raise ValueError('')
        else:
            transforms = [EmptyTransform(), ]
 
        super().__init__(transforms=transforms)


@register
class EmptyTransform(T.Transform):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register
class PadToSize(T.Pad):
    _transformed_types = (
        Image.Image,
        datapoints.Image,
        datapoints.Video,
        datapoints.Mask,
        datapoints.BoundingBoxes,
    )
    
    def __init__(self, canvas_size, fill=0, padding_mode='constant') -> None:
        if isinstance(canvas_size, int):
            canvas_size = (canvas_size, canvas_size)
        self.canvas_size = canvas_size
        self.padding = [0, 0, 0, 0]
        super().__init__(0, fill, padding_mode)

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sz = F.get_canvas_size(flat_inputs[0])
        h, w = self.canvas_size[0] - sz[0], self.canvas_size[1] - sz[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return self._get_params(flat_inputs)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:        
        fill = self._fill[type(inpt)]
        padding = params['padding']
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(self.padding)
        return outputs


@register
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, 
                 max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, 
                 trials: int = 40, p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p 

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        return super().forward(*inputs)


@register
class ConvertBox(T.Transform):
    _transformed_types = (
        datapoints.BoundingBoxes,
    )
    
    def __init__(self, out_fmt='', normalize=False) -> None:
        super().__init__()
        self.out_fmt = out_fmt
        self.normalize = normalize
        self.data_fmt = {
            'xyxy': datapoints.BoundingBoxFormat.XYXY,
            'cxcywh': datapoints.BoundingBoxFormat.CXCYWH
        }

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:  
        if self.out_fmt:
            canvas_size = inpt.canvas_size
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.out_fmt)
            inpt = datapoints.BoundingBoxes(inpt, format=self.data_fmt[self.out_fmt], canvas_size=canvas_size)
        
        if self.normalize:
            inpt = inpt / torch.tensor(inpt.canvas_size[::-1]).tile(2)[None]
        return inpt

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)
