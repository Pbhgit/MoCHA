from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.distributed as dist

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.moec_builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
from .hga import (build_innovative_graph_v3, 
                              innovative_graph_propagation_v3, 
                             )

class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            # self.vision_tower = build_vision_tower(config, delay_load=True)
            # self.mm_projector = build_vision_projector(config)

            self.vision_tower = build_vision_tower(config, load_model = "clip", delay_load=True)
            self.dino_tower = build_vision_tower(config, load_model = "dino", delay_load=True)
            self.siglip_tower = build_vision_tower(config, load_model = "siglip", delay_load=True)
            self.convnext_tower = build_vision_tower(config, load_model = "convnext", delay_load=True)

            self.mm_projector = build_vision_projector(config)
            self.dino_mm_projector = build_vision_projector(config)
            self.siglip_mm_projector = build_vision_projector(config)
            self.convnext_mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )


    def get_vision_tower(self, load_model = "clip"):
        if load_model == "clip":
            vision_tower = getattr(self, 'vision_tower', None)
            if type(vision_tower) is list:
                vision_tower = vision_tower[0]
            return vision_tower
        elif load_model == 'siglip':  
            vision_tower = getattr(self, 'siglip_tower', None)
            if type(vision_tower) is list:
                vision_tower = vision_tower[0]
            return vision_tower
        elif load_model == 'convnext':
            vision_tower = getattr(self, 'convnext_tower', None)
            if type(vision_tower) is list:
                vision_tower = vision_tower[0]
            return vision_tower 
        else:
            vision_tower = getattr(self, 'dino_tower', None)
            if type(vision_tower) is list:
                vision_tower = vision_tower[0]
            return vision_tower


    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type
        pretrain_dino_mm_mlp_adapter = model_args.pretrain_dino_mm_mlp_adapter
        pretrain_convnext_mm_mlp_adapter = model_args.pretrain_convnext_mm_mlp_adapter
        pretrain_siglip_mm_mlp_adapter = model_args.pretrain_siglip_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower
        self.config.scales = model_args.scales

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args, load_model = "clip")
            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        if self.get_vision_tower(load_model='siglip') is None:
            siglip_tower = build_vision_tower(model_args, load_model="siglip")
            if fsdp is not None and len(fsdp) > 0:
                self.siglip_tower = [siglip_tower]
            else:
                self.siglip_tower = siglip_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                siglip_tower = self.siglip_tower[0]
            else:
                siglip_tower = self.siglip_tower
            siglip_tower.load_model()

        if self.get_vision_tower(load_model = "convnext") is None:
            convnext_tower = build_vision_tower(model_args, load_model="convnext")
            if fsdp is not None and len(fsdp) > 0:
                self.convnext_tower = [convnext_tower]
            else:
                self.convnext_tower = convnext_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                convnext_tower = self.convnext_tower[0]
            else:
                convnext_tower = self.convnext_tower
            convnext_tower.load_model()
        

        if self.get_vision_tower(load_model = "dino") is None:
            dino_tower = build_vision_tower(model_args, load_model="dino")
            if fsdp is not None and len(fsdp) > 0:
                self.dino_tower = [dino_tower]
            else:
                self.dino_tower = dino_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                dino_tower = self.dino_tower[0]
            else:
                dino_tower = self.dino_tower
            dino_tower.load_model()

        # if dist.is_initialized():
        #     dist.barrier()
  

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        # self.config.hidden_size = 1280
        # self.config.mm_hidden_size = vision_tower.hidden_size * 2
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type
        self.config.num_experts = model_args.num_experts
        self.config.num_selected = model_args.num_selected
        self.config.num_layers = model_args.num_layers
        self.config.dropout = model_args.dropout
        self.config.mlp_smoe = model_args.mlp_smoe

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
            # print(self.mm_projector)
            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if getattr(self, 'siglip_mm_projector', None) is None:
            self.siglip_mm_projector = build_vision_projector(self.config)
            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.siglip_mm_projector.parameters():
                p.requires_grad = True

        if getattr(self, 'convnext_mm_projector', None) is None:
            self.convnext_mm_projector = build_vision_projector(self.config)
            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.convnext_mm_projector.parameters():
                p.requires_grad = True

        if getattr(self, 'dino_mm_projector', None) is None:
            self.dino_mm_projector = build_vision_projector(self.config)
            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.dino_mm_projector.parameters():
                p.requires_grad = True
################################################################################################################    
        if pretrain_mm_mlp_adapter is not None:
            print("Loading pretrained CLIP adpater!!!")
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            # print("Keys in weight file:", mm_projector_weights.keys())
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}            
            def load_mlpm_moe_state_dict(mlpm_moe, state_dict):
                """
                load MLPMoE's state_dict.
                """
                mlpm_moe.gate.load_state_dict({
                    'weight': state_dict['gate.weight']
                }, strict=False)
                for i, expert in enumerate(mlpm_moe.experts):
                    expert_state_dict = {
                        '0.weight': state_dict[f'experts.{i}.0.weight'],
                        '0.bias': state_dict[f'experts.{i}.0.bias'],
                        '2.weight': state_dict[f'experts.{i}.2.weight'],
                        '2.bias': state_dict[f'experts.{i}.2.bias']
                    }
                    expert.load_state_dict(expert_state_dict, strict=False)
            load_mlpm_moe_state_dict(self.mm_projector, get_w(mm_projector_weights, 'mm_projector'))
            
        if pretrain_siglip_mm_mlp_adapter is not None:
            print("Loading pretrained SIGLIP adpater!!!")
            siglip_mm_projector_weights = torch.load(pretrain_siglip_mm_mlp_adapter, map_location='cpu')
            # print("Keys in weight file:", siglip_mm_projector_weights.keys())
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}            
            def load_mlpm_moe_state_dict(mlpm_moe, state_dict):
                mlpm_moe.gate.load_state_dict({
                    'weight': state_dict['gate.weight']
                }, strict=False)
                for i, expert in enumerate(mlpm_moe.experts):
                    expert_state_dict = {
                        '0.weight': state_dict[f'experts.{i}.0.weight'],
                        '0.bias': state_dict[f'experts.{i}.0.bias'],
                        '2.weight': state_dict[f'experts.{i}.2.weight'],
                        '2.bias': state_dict[f'experts.{i}.2.bias']
                    }
                    expert.load_state_dict(expert_state_dict, strict=False)
            load_mlpm_moe_state_dict(self.siglip_mm_projector, get_w(siglip_mm_projector_weights, 'siglip_mm_projector'))
            
        if pretrain_convnext_mm_mlp_adapter is not None:
            print("Loading pretrained CONVNEXT adpater!!!")
            convnext_mm_projector_weights = torch.load(pretrain_convnext_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            def load_mlpm_moe_state_dict(mlpm_moe, state_dict):
                mlpm_moe.gate.load_state_dict({
                    'weight': state_dict['gate.weight']
                }, strict=False)
                for i, expert in enumerate(mlpm_moe.experts):
                    expert_state_dict = {
                        '0.weight': state_dict[f'experts.{i}.0.weight'],
                        '0.bias': state_dict[f'experts.{i}.0.bias'],
                        '2.weight': state_dict[f'experts.{i}.2.weight'],
                        '2.bias': state_dict[f'experts.{i}.2.bias']
                    }
                    expert.load_state_dict(expert_state_dict, strict=False)
            load_mlpm_moe_state_dict(self.convnext_mm_projector, get_w(convnext_mm_projector_weights, 'convnext_mm_projector'))

        if pretrain_dino_mm_mlp_adapter is not None:
            print("Loading pretrained DINO adpater!!!")
            dino_mm_projector_weights = torch.load(pretrain_dino_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            def load_mlpm_moe_state_dict(mlpm_moe, state_dict):
                mlpm_moe.gate.load_state_dict({
                    'weight': state_dict['gate.weight']
                }, strict=False)
                for i, expert in enumerate(mlpm_moe.experts):
                    expert_state_dict = {
                        '0.weight': state_dict[f'experts.{i}.0.weight'],
                        '0.bias': state_dict[f'experts.{i}.0.bias'],
                        '2.weight': state_dict[f'experts.{i}.2.weight'],
                        '2.bias': state_dict[f'experts.{i}.2.bias']
                    }
                    expert.load_state_dict(expert_state_dict, strict=False)
            load_mlpm_moe_state_dict(self.dino_mm_projector, get_w(dino_mm_projector_weights, 'dino_mm_projector'))           
        

def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format. 
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor



class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_dino_vision_tower(self):
        return self.get_model().get_vision_tower(load_model = "dino")
    
    def get_convnext_vision_tower(self):
        return self.get_model().get_vision_tower(load_model = "convnext")
    
    def get_siglip_vision_tower(self):
        return self.get_model().get_vision_tower(load_model = "siglip")
    
    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features, mlp_balanced_loss_clip,  mlp_router_z_loss_clip = self.get_model().mm_projector(image_features)
        # print("CLIP output shape:", image_features.shape)
        return image_features, mlp_balanced_loss_clip, mlp_router_z_loss_clip
      
    def encode_images_withclip(self, images):
        image_features_clip = self.get_model().get_vision_tower(load_model = "clip")(images)
        image_features_clip, mlp_balanced_loss_clip, mlp_router_z_loss_clip = self.get_model().mm_projector(image_features_clip)
        # print("CLIP output shape:", image_features_clip.shape)
        return image_features_clip, mlp_balanced_loss_clip, mlp_router_z_loss_clip
    
    def encode_images_withsiglip(self, images):
        image_features_siglip = self.get_model().get_vision_tower(load_model = "siglip")(images)
        # print("Siglip output shape:", image_features_siglip.shape)
        image_features_siglip, mlp_balanced_loss_siglip, mlp_router_z_loss_siglip = self.get_model().siglip_mm_projector(image_features_siglip)
       
        return image_features_siglip, mlp_balanced_loss_siglip, mlp_router_z_loss_siglip
    
    def encode_images_withconvnext(self, images):
        image_features_convnext = self.get_model().get_vision_tower(load_model = "convnext")(images)
        # print("convnext output shape:", image_features_convnext.shape)
        avg_pool = nn.AvgPool1d(kernel_size=3, stride=3, padding=0)
        bsz, num_patches, feature_dim = image_features_convnext.shape
        image_features_convnext = avg_pool(image_features_convnext)
        image_features_convnext, mlp_balanced_loss_convnext, mlp_router_z_loss_convnext = self.get_model().convnext_mm_projector(image_features_convnext)
        # print("ConvNext output shape:", image_features_convnext.shape)
        return image_features_convnext, mlp_balanced_loss_convnext, mlp_router_z_loss_convnext
    
    def encode_images_withdino(self, images):
        image_features_dino = self.get_model().get_vision_tower(load_model = "dino")(images)
        image_features_dino, mlp_balanced_loss_dino, mlp_router_z_loss_dino = self.get_model().dino_mm_projector(image_features_dino)
        # print("Dino output shape:", image_features_dino.shape)
        return image_features_dino, mlp_balanced_loss_dino, mlp_router_z_loss_dino
    
    # clip siglip dino convext token concatenation 
    def prepare_inputs_labels_for_multimodal_interleaved_withsiglipdinoconvnext(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        mlp_balanced_loss = None
        mlp_router_z_loss = None
        vision_tower = self.get_vision_tower()
        # print("images is: ", images.shape)
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, mlp_balanced_loss, mlp_router_z_loss
        
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)

            image_features_clip, mlp_balanced_loss_clip, mlp_router_z_loss_clip = self.encode_images_withclip(concat_images)
            image_features_siglip, mlp_balanced_loss_siglip, mlp_router_z_loss_siglip = self.encode_images_withsiglip(concat_images)
            image_features_dino, mlp_balanced_loss_dino, mlp_router_z_loss_dino = self.encode_images_withdino(concat_images)
            image_features_convnext, mlp_balanced_loss_convnext, mlp_router_z_loss_convnext = self.encode_images_withconvnext(concat_images)

            split_sizes = [image.shape[0] for image in images]
            image_features_clip = torch.split(image_features_clip, split_sizes, dim=0)
            image_features_siglip = torch.split(image_features_siglip, split_sizes, dim=0)
            image_features_dino = torch.split(image_features_dino, split_sizes, dim=0)
            image_features_convnext = torch.split(image_features_convnext, split_sizes, dim=0)

            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                # image_features = [x.flatten(0, 1) for x in image_features]
                image_features_clip = [x.flatten(0, 1) for x in image_features_clip]
                image_features_siglip = [x.flatten(0, 1) for x in image_features_siglip]
                image_features_dino = [x.flatten(0, 1) for x in image_features_dino]
                image_features_convnext = [x.flatten(0, 1) for x in image_features_convnext]
            elif mm_patch_merge_type.startswith('spatial'):
                # new_image_features = []
                new_image_features_clip = []
                new_image_features_siglip = []
                new_image_features_dino = []
                new_image_features_convnext = []
                for image_idx, (clip_feature, siglip_feature, dino_feature, convnext_feature) in enumerate(zip(image_features_clip, image_features_siglip, image_features_dino, image_features_convnext)):
                    # if image_feature.shape[0] > 1:
                    #     base_image_feature = image_feature[0]
                    #     image_feature = image_feature[1:]
                    if clip_feature.shape[0] > 1:
                        base_clip_feature = clip_feature[0]
                        clip_feature = clip_feature[1:]

                        base_siglip_feature = siglip_feature[0]
                        siglip_feature = siglip_feature[1:]
    
                        base_dino_feature = dino_feature[0]
                        dino_feature = dino_feature[1:]

                        base_convnext_feature = convnext_feature[0]
                        convnext_feature = convnext_feature[1:]

                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_clip_feature.shape[0]

                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                                image_sizes[image_idx], self.config.image_grid_pinpoints,
                                self.get_vision_tower().config.image_size
                            )
                            clip_feature = clip_feature.view(num_patch_height, num_patch_width, height, width, -1)
                            siglip_feature = siglip_feature.view(num_patch_height, num_patch_width, height, width, -1)
                            convnext_feature = convnext_feature.view(num_patch_height, num_patch_width, height, width, -1)
                            dino_feature = dino_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError

                        if 'unpad' in mm_patch_merge_type:
                            clip_feature = clip_feature.permute(4, 0, 2, 1, 3).contiguous()
                            clip_feature = clip_feature.flatten(1, 2).flatten(2, 3)
                            clip_feature = unpad_image(clip_feature, image_sizes[image_idx])
                            clip_feature = torch.cat((
                                clip_feature,
                                self.model.image_newline[:, None, None].expand(*clip_feature.shape[:-1], 1).to(clip_feature.device)
                            ), dim=-1)
                            clip_feature = clip_feature.flatten(1, 2).transpose(0, 1)

                            siglip_feature = siglip_feature.permute(4, 0, 2, 1, 3).contiguous()
                            siglip_feature = siglip_feature.flatten(1, 2).flatten(2, 3)
                            siglip_feature = unpad_image(siglip_feature, image_sizes[image_idx])
                            siglip_feature = torch.cat((
                                siglip_feature,
                                self.model.image_newline[:, None, None].expand(*siglip_feature.shape[:-1], 1).to(siglip_feature.device)
                            ), dim=-1)
                            siglip_feature = siglip_feature.flatten(1, 2).transpose(0, 1)

                            dino_feature = dino_feature.permute(4, 0, 2, 1, 3).contiguous()
                            dino_feature = dino_feature.flatten(1, 2).flatten(2, 3)
                            dino_feature = unpad_image(dino_feature, image_sizes[image_idx])
                            dino_feature = torch.cat((
                                dino_feature,
                                self.model.image_newline[:, None, None].expand(*dino_feature.shape[:-1], 1).to(dino_feature.device)
                            ), dim=-1)
                            dino_feature = dino_feature.flatten(1, 2).transpose(0, 1)

                            convnext_feature = convnext_feature.permute(4, 0, 2, 1, 3).contiguous()
                            convnext_feature = convnext_feature.flatten(1, 2).flatten(2, 3)
                            convnext_feature = unpad_image(convnext_feature, image_sizes[image_idx])
                            convnext_feature = torch.cat((
                                convnext_feature,
                                self.model.image_newline[:, None, None].expand(*convnext_feature.shape[:-1], 1).to(convnext_feature.device)
                            ), dim=-1)
                            convnext_feature = convnext_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            siglip_feature = siglip_feature.permute(0, 2, 1, 3, 4).contiguous()
                            siglip_feature = siglip_feature.flatten(0, 3)

                            clip_feature = clip_feature.permute(0, 2, 1, 3, 4).contiguous()
                            clip_feature = clip_feature.flatten(0, 3)

                            convnext_feature = convnext_feature.permute(0, 2, 1, 3, 4).contiguous()
                            convnext_feature = convnext_feature.flatten(0, 3)

                            dino_feature = dino_feature.permute(0, 2, 1, 3, 4).contiguous()
                            dino_feature = dino_feature.flatten(0, 3)
                        clip_feature = torch.cat((base_clip_feature, clip_feature), dim=0)
                        siglip_feature = torch.cat((base_siglip_feature, siglip_feature), dim=0)
                        convnext_feature = torch.cat((base_convnext_feature, convnext_feature), dim=0)
                        dino_feature = torch.cat((base_dino_feature, dino_feature), dim=0)
                    else:
                        clip_feature = clip_feature[0]
                        siglip_feature = siglip_feature[0]
                        dino_feature = dino_feature[0]
                        convnext_feature = convnext_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            clip_feature = torch.cat((
                                clip_feature,
                                self.model.image_newline[None].to(clip_feature.device)
                            ), dim=0)

                            siglip_feature = torch.cat((
                                siglip_feature,
                                self.model.image_newline[None].to(siglip_feature.device)
                            ), dim=0)

                            dino_feature = torch.cat((
                                dino_feature,
                                self.model.image_newline[None].to(dino_feature.device)
                            ), dim=0)

                            convnext_feature = torch.cat((
                                convnext_feature,
                                self.model.image_newline[None].to(convnext_feature.device)
                            ), dim=0)
                    new_image_features_clip.append(clip_feature)
                    new_image_features_siglip.append(siglip_feature)
                    new_image_features_dino.append(dino_feature)
                    new_image_features_convnext.append(convnext_feature)
                image_features_clip = new_image_features_clip
                image_features_siglip = new_image_features_siglip
                image_features_dino = new_image_features_dino
                image_features_convnext = new_image_features_convnext
                
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:           
            image_features_clip, mlp_balanced_loss_clip, mlp_router_z_loss_clip = self.encode_images_withclip(images)
            image_features_siglip, mlp_balanced_loss_siglip, mlp_router_z_loss_siglip = self.encode_images_withsiglip(images)
            image_features_dino, mlp_balanced_loss_dino, mlp_router_z_loss_dino = self.encode_images_withdino(images)
            image_features_convnext, mlp_balanced_loss_convnext, mlp_router_z_loss_convnext = self.encode_images_withconvnext(images)

        
        mlp_balanced_loss = (
            mlp_balanced_loss_clip
            + mlp_balanced_loss_siglip
            + mlp_balanced_loss_dino
            + mlp_balanced_loss_convnext
        )
        mlp_router_z_loss = (
            mlp_router_z_loss_clip
            + mlp_router_z_loss_siglip
            + mlp_router_z_loss_dino
            + mlp_router_z_loss_convnext
        )

        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_input_embeds = torch.cat([self.get_model().embed_tokens(cur_input_ids), torch.empty(0, dtype=self.get_model().embed_tokens(cur_input_ids).dtype, device=self.device)], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features_clip = image_features_clip[cur_image_idx] # [num_patches, feature_dim]
                    # print("cur_image_features_clip shape:",cur_image_features_clip.shape)
                    cur_image_features_siglip = image_features_siglip[cur_image_idx] 
                    cur_image_features_dino = image_features_dino[cur_image_idx]
                    cur_image_features_convnext = image_features_convnext[cur_image_idx]
                    cur_image_idx += 1
                    num_patches = cur_image_features_clip.shape[0]
                    clip_dim = cur_image_features_clip.shape[1]
                    siglip_num_patches = cur_image_features_siglip.shape[0]
                    convnext_num_patches = cur_image_features_convnext.shape[0]
                    dino_num_patches = cur_image_features_dino.shape[0]
             
                    # HGA attention merge features
                    merged_features = torch.cat((cur_image_features_siglip, cur_image_features_convnext, cur_image_features_clip, cur_image_features_dino), dim=0)
                    group_idx_ = torch.tensor(
                            [0]*siglip_num_patches + [1]*convnext_num_patches + [2]*num_patches + [3]*dino_num_patches,
                            device=merged_features.device
                    )
                    edge_index, edge_weight = build_innovative_graph_v3(
                        merged_features, group_idx_, k=10, cross_weight=0.7
                    )
                    merged_features_ = innovative_graph_propagation_v3(
                        merged_features,
                        edge_index, edge_weight,
                    )
                                
                    # merged features
                    # merged_features = torch.cat((cur_image_features_siglip, cur_image_features_dino, cur_image_features_convnext, cur_image_features_clip), dim=0)

                
                    cur_new_input_embeds.append(merged_features_)
                    cur_new_labels.append(torch.full((merged_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        # print(tokenizer_model_max_length)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        
        # print("new_input_embeds is: ", new_input_embeds)
        # print("mlp_balanced_loss_clip is: ", mlp_balanced_loss_clip)
        # print("mlp_balanced_loss is: ", mlp_balanced_loss)
        # print("mlp_router_z_loss is: ", mlp_router_z_loss)

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, mlp_balanced_loss, mlp_router_z_loss
    
    
    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False   
