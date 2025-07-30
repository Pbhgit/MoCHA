import torch
from open_clip import create_model_from_pretrained 
from timm.models.eva import Eva
import torch.nn as nn
import torch.nn.functional as F
from .base_encoder import BaseVisionTower
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from open_clip import create_model_from_pretrained, get_tokenizer
from ezcolorlog import root_logger as logger

# TODO: move elsewhere?
IS_XLA_AVAILABLE = False
try:
    import torch_xla
    IS_XLA_AVAILABLE = True
except ImportError:
    pass

class ProcessorWrapper:
    def __init__(self, transform, height=378, width=378, image_mean = [0.48145466, 0.4578275, 0.40821073]):
        self._crop_size = {
            "height": height,
            "width": width,
        }
        self._transforms = transform
        #print(transform)
        self.image_mean = image_mean

    @property
    def crop_size(self):
        return self._crop_size

    def preprocess(self, image, return_tensors='pt'):
        # Ensure image is a PIL Image
        output = {}
        output['pixel_values'] = [self._transforms(image)]
        return output
    
def extract_interp(model_name):
    interp = None
    base_model_name = model_name

    if "interp" in model_name:
        base_model_name = model_name.split('-interp')[0]

    parts = model_name.split("-")
    for part in parts:
        if part.startswith("interp"):
            interp = int(part[6:])

    return base_model_name, interp


class ClipVisionTower(BaseVisionTower):
    def __init__(self, vision_tower_name, args, delay_load=False):
        super(ClipVisionTower, self).__init__(vision_tower_name, args, delay_load)
        base_model_name, interp = extract_interp(vision_tower_name)
        self.vision_tower_name = base_model_name
        self._interp_size = interp 
        if not self.delay_load:
            self.load_model()
        elif self.unfreeze_mm_vision_tower:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            logger.debug(f"{self.vision_tower_name} is already loaded, `load_model` called again, skipping.")
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)

        self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)
        self.is_loaded = True

        if IS_XLA_AVAILABLE:
            # Very Important for TorchXLA
            from torch_xla.utils.checkpoint import checkpoint
            self.vision_tower.vision_model.encoder._gradient_checkpointing_func = checkpoint

    def _feature_select(self, image_features):
        if self.select_feature == 'patch':
            features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return features

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        return self._feature_select(image_features)

    def interpolate(self, image_features):
        if self._interp_size is None:
            return image_features

        b, num_tokens, dim = image_features.shape

        if num_tokens != self.num_patches:
            target_h = target_w = int(self._interp_size ** 0.5)
            h = w = int(num_tokens ** 0.5)

            image_features = image_features.view(b, h, w, dim)
            image_features = image_features.permute(0, 3, 1, 2).contiguous()

            image_features = F.interpolate(
                image_features.to(torch.float32),
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            ).to(image_features.dtype)

            # Permute the dimensions back to (b, target_h, target_w, dim)
            image_features = image_features.permute(0, 2, 3, 1).contiguous()

            # Flatten the spatial dimensions (target_h, target_w) into a single dimension
            image_features = image_features.flatten(1, 2)

        return image_features

    def _forward(self, images):
        if IS_XLA_AVAILABLE:
            from torch_xla.utils.checkpoint import checkpoint
            self.vision_tower.vision_model.encoder._gradient_checkpointing_func = checkpoint

        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            interp_features = self.interpolate(image_features)
            return interp_features
        

class EvaClipVisionTower(ClipVisionTower):
    def __init__(self, vision_tower_name, args, delay_load=False):
        super(ClipVisionTower, self).__init__(vision_tower_name, args, delay_load)
        if not self.delay_load:
            self.load_model()
        elif self.unfreeze_mm_vision_tower:
            self.load_model()

    def load_model(self, device_map=None):
        if self.vision_tower_name in (
            "eva/CLIP-ViT-L-336",
            "timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k"
        ):
            self.vision_model = "evaclip"
            clip_model, processor = create_model_from_pretrained('hf-hub:timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k')
            self.image_processor = ProcessorWrapper(processor, height=336, width=336)
            self._patch_size = 14
        elif self.vision_tower_name in (
            "eva/CLIP-ViT-L-224",
            "timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k"
        ):
            self.vision_model = "evaclip"
            clip_model, processor = create_model_from_pretrained('hf-hub:timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k')
            self.image_processor = ProcessorWrapper(processor, height=224, width=224)
            self._patch_size = 14
        else:
            raise ValueError(f'Unknown vision tower: {self.vision_tower_name}')

        self.vision_tower: Eva = clip_model.visual.trunk
        self.vision_tower.output_tokens = True
        self._hidden_size = 1024

        self._image_size = self.vision_tower.pretrained_cfg["input_size"][-1]

        self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)
        self.is_loaded = True

    def _forward(self, images):
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            image_forward_outs = self.vision_tower.forward_features(images.to(device=self.device, dtype=self.dtype))
            image_features = self._feature_select(image_forward_outs).to(images.dtype)

            return image_features