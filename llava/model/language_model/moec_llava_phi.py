from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from dataclasses import dataclass, field
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from .phi.configuration_phi import PhiConfig
from .phi.modeling_phi import PhiModel, PhiForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..moec_llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
import torch.distributed as dist

class LlavaPhiConfig(PhiConfig):
    model_type = "llava_phi"
    balance_loss_coef: float = 0.1
    router_z_loss_coef: float = 0.01
    mlp_smoe: bool = True
    local_rank = None

class LlavaPhiModel(LlavaMetaModel, PhiModel):
    config_class = LlavaPhiConfig

    def __init__(self, config: PhiConfig):
        super(LlavaPhiModel, self).__init__(config)


class LlavaPhiForCausalLM(PhiForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaPhiConfig  
   
    def __init__(self, config):
        super(PhiForCausalLM, self).__init__(config)
        self.model = LlavaPhiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)


        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # import ipdb
        # ipdb.set_trace()
        # print(f'rank {dist.get_rank()}', 'before prepare_inputs_labels_for_multimodal')

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels, 
                mlp_balance_loss,
                mlp_router_z_loss              
            ) = self.prepare_inputs_labels_for_multimodal_interleaved_withsiglipdinoconvnext(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
            
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            if self.config.mlp_smoe:
                if self.config.local_rank == 0:
                    print('language loss: ', loss.item())
                mlp_balance_loss = mlp_balance_loss.sum(dim=-1).mean()
                mlp_balance_loss = self.config.balance_loss_coef * mlp_balance_loss
                loss += mlp_balance_loss

                mlp_router_z_loss = mlp_router_z_loss.sum(dim=-1).mean()
                mlp_router_z_loss = self.config.router_z_loss_coef * mlp_router_z_loss
                loss += mlp_router_z_loss
                # print('mlp balance loss: ', mlp_balance_loss.item(), 'mlp router z loss: ', mlp_router_z_loss.item())

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        # dist.barrier()
        # print(f'rank {dist.get_rank()}', 'after prepare_inputs_labels_for_multimodal')
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                _, 
                _
            ) = self.prepare_inputs_labels_for_multimodal_interleaved_withsiglipdinoconvnext(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            # print(self.get_model())
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs  # dict
    
AutoConfig.register("llava_phi", LlavaPhiConfig)
# AutoTokenizer.register(LlavaPhiConfig, PhiTokenizer)
AutoModelForCausalLM.register(LlavaPhiConfig, LlavaPhiForCausalLM)
