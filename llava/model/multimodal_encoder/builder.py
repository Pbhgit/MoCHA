import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .dino_encoder import DINOVisionTower
from .convnext_encoder import ConvNextVisionTower
from .siglip_encoder import SigLipVisionTower
from .eva_clip_encoder import EvaClipVisionTower

def build_vision_tower(vision_tower_cfg, load_model="clip", **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)

    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if load_model == "clip":            
            if use_s2:
                return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
            else:
                return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        elif load_model == 'convnext':
            return ConvNextVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        elif load_model == 'siglip':
            return SigLipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        elif load_model == 'evaclip':
            return EvaClipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        else:           
            return DINOVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    


   
    raise ValueError(f'Unknown vision tower: {vision_tower}')