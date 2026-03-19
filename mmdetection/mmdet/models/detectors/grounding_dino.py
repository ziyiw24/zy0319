# Copyright (c) OpenMMLab. All rights reserved.
import copy
import re
import warnings
from collections import Counter, defaultdict
from typing import Dict, Optional, Tuple, Union
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.runner.amp import autocast
from torch import Tensor
from mmcv.ops import roi_align
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.structures.bbox import bbox2roi, get_box_tensor
from mmdet.utils import ConfigType
from ..layers import SinePositionalEncoding
from ..layers.transformer.grounding_dino_layers import (
    GroundingDinoTransformerDecoder, GroundingDinoTransformerEncoder)
from .dino import DINO
from .glip import (create_positive_map, create_positive_map_label_to_token,
                   run_ner)


def clean_label_name(name: str) -> str:
    name = re.sub(r'\(.*\)', '', name)
    name = re.sub(r'_', ' ', name)
    name = re.sub(r'  ', ' ', name)
    return name


def chunks(lst: list, n: int) -> list:
    """Yield successive n-sized chunks from lst."""
    all_ = []
    for i in range(0, len(lst), n):
        data_index = lst[i:i + n]
        all_.append(data_index)
    counter = 0
    for i in all_:
        counter += len(i)
    assert (counter == len(lst))

    return all_


@MODELS.register_module()
class GroundingDINO(DINO):
    """Implementation of `Grounding DINO: Marrying DINO with Grounded Pre-
    Training for Open-Set Object Detection.

    <https://arxiv.org/abs/2303.05499>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/GroundingDINO>`_.
    """

    def __init__(self,
                 language_model,
                 *args,
                 use_autocast=False,
                 loss_align_weight=0.2,
                 prompt_template: str = None,
                 prompt_template_map: Optional[Dict[str, str]] = None,
                 prompt_attrs: Optional[Dict[str, Union[str, Dict[str, str]]]] = None,
                 prompt_domain: str = None,
                 max_prompt_classes: int = 20,
                 dataset_name_key: str = 'dataset',
                 negative_prompt_template: Optional[str] = None,
                 class_name_corrections: Optional[Dict[str, str]] = None,
                 **kwargs) -> None:

        self.language_model_cfg = language_model
        self._special_tokens = '. '
        self.use_autocast = use_autocast
        self.loss_align_weight = loss_align_weight
        self.prompt_template = prompt_template
        self.prompt_template_map = prompt_template_map or {}
        self.prompt_attrs = prompt_attrs or {}
        self.prompt_domain = prompt_domain
        self.max_prompt_classes = max_prompt_classes
        self.dataset_name_key = dataset_name_key
        self.negative_prompt_template = negative_prompt_template
        # Map class names to better text for BERT (e.g. "seacucumber" -> "sea cucumber")
        self.class_name_corrections = class_name_corrections or {}
        super().__init__(*args, **kwargs)

    def _get_dataset_name(self, data_samples) -> Optional[str]:
        """Attempt to extract a dataset identifier from the sample metainfo."""
        meta = getattr(data_samples, 'metainfo', None)
        if not isinstance(meta, dict):
            return None
        # Common keys used by datasets for the dataset name
        for k in (self.dataset_name_key, 'dataset_name', 'dataset', 'task'):
            if k in meta and isinstance(meta[k], str):
                return meta[k]
        return None

    def _get_prompt_template(self, dataset_name: Optional[str] = None) -> str:
        """Resolve the prompt template based on dataset name (if provided)."""
        if dataset_name and dataset_name in self.prompt_template_map:
            return self.prompt_template_map[dataset_name]
        if self.prompt_template:
            return self.prompt_template
        return "a photo of a {class}"

    def _get_prompt_attrs(self, dataset_name: Optional[str] = None) -> Dict[str, str]:
        """Build prompt attributes dict (color/shape/etc) from config."""
        if not isinstance(self.prompt_attrs, dict):
            return {}
        # If prompt_attrs is nested per-dataset, use that first.
        if dataset_name and dataset_name in self.prompt_attrs:
            return self.prompt_attrs[dataset_name] or {}
        # Otherwise treat prompt_attrs as a flat dict of attributes.
        return self.prompt_attrs or {}

    def _make_prompt(self,
                     class_name: str,
                     dataset_name: Optional[str] = None,
                     attrs: Optional[Dict[str, str]] = None) -> str:
        """Convert a raw class name into a more descriptive prompt.

        This supports:
          - `prompt_template` / `prompt_template_map` for per-dataset templates
          - `prompt_attrs` to inject attributes like color/shape/scene
          - `prompt_domain` as an optional extra field
        """
        # Apply name corrections first (e.g. "seacucumber" -> "sea cucumber")
        class_name = self.class_name_corrections.get(class_name, class_name)
        class_name = class_name.replace('_', ' ')
        template = self._get_prompt_template(dataset_name)
        # Build formatting context
        context = defaultdict(str)
        context['class'] = class_name
        context['domain'] = self.prompt_domain or ''
        if attrs:
            context.update(attrs)
        try:
            prompt = template.format_map(context)
        except Exception:
            prompt = f"a photo of a {class_name}"

        # Optionally append a negative prompt to encourage more discriminative
        # grounding in few-shot / cross-domain settings.
        if self.negative_prompt_template:
            try:
                negative_prompt = self.negative_prompt_template.format_map(context)
            except Exception:
                negative_prompt = ''
            if negative_prompt:
                prompt = f"{prompt} {negative_prompt}"

        return prompt

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = GroundingDinoTransformerEncoder(**self.encoder)
        self.decoder = GroundingDinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        # text modules
        self.language_model = MODELS.build(self.language_model_cfg)
        self.text_feat_map = nn.Linear(
            self.language_model.language_backbone.body.language_dim,
            self.embed_dims,
            bias=True)

        # PGTA: prototype-guided text alignment (few-shot visual prototype → text)
        self.prototype_dim = 256
        self.prototype_memory = nn.Parameter(
            torch.zeros(self.bbox_head.num_classes, self.prototype_dim),
            requires_grad=False)
        self.prototype_proj = nn.Linear(
            self.prototype_dim,
            self.embed_dims)
        # Prototype-guided query ranking (reliability-aware).
        # Use a fixed lambda to avoid relying on gradients through top-k.
        self.proto_lambda = 0.3
        # Adaptive reliability threshold: small-class few-shot datasets saturate at lower counts.
        # For K<=5 (e.g. UODD 3-class): threshold=8 -> reliability reaches 1.0 by epoch 8.
        # For K>10: threshold=20 (original), which suits larger datasets.
        _n_cls = self.bbox_head.num_classes
        self.proto_reliability_threshold = float(max(8, min(20, _n_cls * 3)))
        # Query adapter: residual MLP for learned prototype modulation (decoder stage).
        self.proto_query_adapter_scale = 0.2
        self.proto_query_adapter = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        # Learnable scale (init conservative) + per-class reliability tracking
        # Note: for extremely low-iter regimes (e.g., 1-shot, 1 iter/epoch),
        # an overly small init can make PGTA effectively inactive.
        self.pgta_alpha = nn.Parameter(torch.tensor(-2.0))  # sigmoid -> ~0.12
        self.register_buffer(
            'prototype_update_counts',
            torch.zeros(self.bbox_head.num_classes, dtype=torch.long),
            persistent=True)
        # Only apply prototypes that have been updated enough times.
        # Use a lower threshold for small-class datasets to avoid PGTA being delayed.
        # Fix A3: Require at least 3 per-class updates before PGTA activates.
        # This avoids noisy prototypes corrupting queries in the first epoch.
        self.pgta_min_updates = 3 if self.bbox_head.num_classes <= 5 else 5
        # Fix C: 2-phase training — DCPA activates only when each class has >= 5 updates.
        # Phase 1 (epoch 1-~2): pure GroundingDINO fine-tuning.
        # Phase 2 (epoch ~3+): PGTA + DCPA with mature prototypes.
        self.dcpa_min_per_class = 5
        # Projection for prototype update: ROI feature (backbone) -> detection feature space.
        self.proto_feat_proj = nn.Linear(self.prototype_dim, self.prototype_dim)

        # DCPA: prototype -> text space (for alignment with text semantic space)
        self.proto_to_text = nn.Linear(self.prototype_dim, self.embed_dims)
        # Token-level cross-attn: t' = t + alpha * Attention(Q=text, K=V=proto)
        # proto_text_adapter produces value from proto for cross-attn.
        self.proto_text_adapter = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        self.dcpa_alpha_raw = nn.Parameter(torch.tensor(-2.0))  # sigmoid -> ~0.12, allows visible text adaptation
        self.dcpa_warmup_threshold = 20  # only apply when prototype_update_counts.sum() > 20
        self.dcpa_attn_temperature = 0.07  # sharper attention for few-shot
        self.dcpa_dropout_prob = 0.3  # prototype dropout in training

        # self.learnable_parameters = nn.Parameter(torch.randn(1, 256))
        
    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        nn.init.constant_(self.text_feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.text_feat_map.weight.data)
        if hasattr(self, 'prototype_proj'):
            nn.init.xavier_uniform_(self.prototype_proj.weight.data)
            if self.prototype_proj.bias is not None:
                nn.init.zeros_(self.prototype_proj.bias.data)
        if hasattr(self, 'proto_query_adapter'):
            for m in self.proto_query_adapter.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        if hasattr(self, 'proto_feat_proj'):
            nn.init.xavier_uniform_(self.proto_feat_proj.weight.data)
            if self.proto_feat_proj.bias is not None:
                nn.init.zeros_(self.proto_feat_proj.bias.data)
        if hasattr(self, 'proto_to_text'):
            nn.init.xavier_uniform_(self.proto_to_text.weight.data)
            if self.proto_to_text.bias is not None:
                nn.init.zeros_(self.proto_to_text.bias.data)
        if hasattr(self, 'proto_text_adapter'):
            for m in self.proto_text_adapter.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def to_enhance_text_prompts(self, original_caption, enhanced_text_prompts):
        caption_string = ''
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            if word in enhanced_text_prompts:
                enhanced_text_dict = enhanced_text_prompts[word]
                if 'prefix' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['prefix']
                start_i = len(caption_string)
                if 'name' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['name']
                else:
                    caption_string += word
                end_i = len(caption_string)
                tokens_positive.append([[start_i, end_i]])

                if 'suffix' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['suffix']
            else:
                tokens_positive.append(
                    [[len(caption_string),
                      len(caption_string) + len(word)]])
                caption_string += word
            caption_string += self._special_tokens
        return caption_string, tokens_positive

    def to_plain_text_prompts(self, original_caption):
        caption_string = ''
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            tokens_positive.append(
                [[len(caption_string),
                  len(caption_string) + len(word)]])
            caption_string += word
            caption_string += self._special_tokens
        return caption_string, tokens_positive

    def get_tokens_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompts: Optional[ConfigType] = None
    ) -> Tuple[dict, str, list]:
        """Get the tokens positive and prompts for the caption."""
        if isinstance(original_caption, (list, tuple)) or custom_entities:
            if custom_entities and isinstance(original_caption, str):
                original_caption = original_caption.strip(self._special_tokens)
                original_caption = original_caption.split(self._special_tokens)
                original_caption = list(
                    filter(lambda x: len(x) > 0, original_caption))

            original_caption = [clean_label_name(i) for i in original_caption]

            if custom_entities and enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption, enhanced_text_prompts)
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption)

            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [caption_string],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')
            entities = original_caption
        else:
            if not original_caption.endswith('.'):
                original_caption = original_caption + self._special_tokens
            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [original_caption],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')
            tokens_positive, noun_phrases = run_ner(original_caption)
            entities = noun_phrases
            caption_string = original_caption

        return tokenized, caption_string, tokens_positive, entities

    def get_positive_map(self, tokenized, tokens_positive):
        positive_map = create_positive_map(
            tokenized,
            tokens_positive,
            max_num_entities=self.bbox_head.cls_branches[
                self.decoder.num_layers].max_text_len)
        positive_map_label_to_token = create_positive_map_label_to_token(
            positive_map, plus=1)
        return positive_map_label_to_token, positive_map

    def _build_token_class_mask(self, batch_data_samples, num_tokens, valid_class_indices, device):
        if not self.training or batch_data_samples is None or len(batch_data_samples) == 0:
            return None
        K = self.bbox_head.num_classes
        K_valid = valid_class_indices.size(0)
        B = len(batch_data_samples)
        masks = []
        for b in range(B):
            ds = batch_data_samples[b]
            if not hasattr(ds.gt_instances, 'positive_maps') or not hasattr(ds.gt_instances, 'labels'):
                masks.append(torch.ones(num_tokens, K_valid, device=device, dtype=torch.float32))
                continue
            pos_map = ds.gt_instances.positive_maps
            labels = ds.gt_instances.labels
            if pos_map.dim() == 1:
                pos_map = pos_map.unsqueeze(0)
            T_map = pos_map.size(1)
            T_use = min(num_tokens, T_map)
            token_to_class = torch.zeros(T_use, K, device=device, dtype=torch.float32)
            for c in range(K):
                rows = (labels.to(device) == c).nonzero(as_tuple=True)[0]
                if rows.numel() > 0:
                    token_to_class[:, c] = (pos_map.to(device)[rows].sum(dim=0) > 0).float()[:T_use]
            mask_b = token_to_class[:, valid_class_indices]
            # Tokens with no class (e.g. "."): allow all prototypes to avoid softmax NaN
            row_sum = mask_b.sum(dim=1, keepdim=True)
            mask_b = torch.where(row_sum > 0, mask_b, torch.ones_like(mask_b))
            if T_use < num_tokens:
                # Padding tokens: allow all prototypes to avoid softmax(-inf) -> NaN
                pad = torch.ones(num_tokens - T_use, K_valid, device=device, dtype=torch.float32)
                mask_b = torch.cat([mask_b, pad], dim=0)
            masks.append(mask_b)
        return torch.stack(masks, dim=0)

    def get_tokens_positive_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompt: Optional[ConfigType] = None,
        tokens_positive: Optional[list] = None,
    ) -> Tuple[dict, str, Tensor, list]:
        """Get the tokens positive and prompts for the caption.

        Args:
            original_caption (str): The original caption, e.g. 'bench . car .'
            custom_entities (bool, optional): Whether to use custom entities.
                If ``True``, the ``original_caption`` should be a list of
                strings, each of which is a word. Defaults to False.

        Returns:
            Tuple[dict, str, dict, str]: The dict is a mapping from each entity
            id, which is numbered from 1, to its positive token id.
            The str represents the prompts.
        """
        if tokens_positive is not None:
            if tokens_positive == -1:
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                return None, original_caption, None, original_caption
            else:
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                tokenized = self.language_model.tokenizer(
                    [original_caption],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                positive_map_label_to_token, positive_map = \
                    self.get_positive_map(tokenized, tokens_positive)

                entities = []
                for token_positive in tokens_positive:
                    instance_entities = []
                    for t in token_positive:
                        instance_entities.append(original_caption[t[0]:t[1]])
                    entities.append(' / '.join(instance_entities))
                return positive_map_label_to_token, original_caption, \
                    positive_map, entities

        chunked_size = self.test_cfg.get('chunked_size', -1)
        if not self.training and chunked_size > 0:
            assert isinstance(original_caption,
                              (list, tuple)) or custom_entities is True
            all_output = self.get_tokens_positive_and_prompts_chunked(
                original_caption, enhanced_text_prompt)
            positive_map_label_to_token, \
                caption_string, \
                positive_map, \
                entities = all_output
        else:
            tokenized, caption_string, tokens_positive, entities = \
                self.get_tokens_and_prompts(
                    original_caption, custom_entities, enhanced_text_prompt)
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive)
        return positive_map_label_to_token, caption_string, \
            positive_map, entities

    def get_tokens_positive_and_prompts_chunked(
            self,
            original_caption: Union[list, tuple],
            enhanced_text_prompts: Optional[ConfigType] = None):
        chunked_size = self.test_cfg.get('chunked_size', -1)
        original_caption = [clean_label_name(i) for i in original_caption]

        original_caption_chunked = chunks(original_caption, chunked_size)
        ids_chunked = chunks(
            list(range(1,
                       len(original_caption) + 1)), chunked_size)

        positive_map_label_to_token_chunked = []
        caption_string_chunked = []
        positive_map_chunked = []
        entities_chunked = []

        for i in range(len(ids_chunked)):
            if enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption_chunked[i], enhanced_text_prompts)
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption_chunked[i])
            tokenized = self.language_model.tokenizer([caption_string],
                                                      return_tensors='pt')
            if tokenized.input_ids.shape[1] > self.language_model.max_tokens:
                warnings.warn('Inputting a text that is too long will result '
                              'in poor prediction performance. '
                              'Please reduce the --chunked-size.')
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive)

            caption_string_chunked.append(caption_string)
            positive_map_label_to_token_chunked.append(
                positive_map_label_to_token)
            positive_map_chunked.append(positive_map)
            entities_chunked.append(original_caption_chunked[i])

        return positive_map_label_to_token_chunked, \
            caption_string_chunked, \
            positive_map_chunked, \
            entities_chunked

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        text_dict: Dict,
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        # DCPA: token-level cross-modal alignment t' = t + alpha * Attention(Q=text, K=V=proto)
        text_embed = text_dict['embedded']
        text_embed_original = text_embed.clone()
        counts = getattr(self, 'prototype_update_counts', None)
        # Fix C: Use per-class minimum updates for DCPA warmup (2-phase training).
        dcpa_min_pc = getattr(self, 'dcpa_min_per_class', 5)
        if counts is not None:
            _active_counts = counts[counts > 0]
            has_warmup = _active_counts.numel() > 0 and int(_active_counts.min().item()) >= dcpa_min_pc
        else:
            has_warmup = False
        has_valid_proto = (
            hasattr(self, 'prototype_memory') and self.prototype_memory is not None
            and counts is not None and int(counts.sum().item()) > 0
        )
        drop_dcpa = self.training and getattr(self, 'dcpa_dropout_prob', 0.0) > 0 and (torch.rand(1).item() < getattr(self, 'dcpa_dropout_prob', 0.0))
        if has_warmup and has_valid_proto and hasattr(self, 'proto_to_text') and not drop_dcpa:
            proto_mem = self.prototype_memory.to(
                device=text_embed.device, dtype=text_embed.dtype)
            valid = (counts.to(device=text_embed.device) > 0) & (
                proto_mem.norm(p=2, dim=-1) >= 1e-6)
            if valid.any():
                # proto_to_text: prototype -> text semantic space
                proto_feat = self.proto_to_text(proto_mem[valid])  # (K_valid, C)
                proto_value = self.proto_text_adapter(proto_feat)  # (K_valid, C)
                # Normalize for stable attention (cosine similarity)
                proto_feat_n = F.normalize(proto_feat, p=2, dim=-1, eps=1e-6)
                text_for_attn = F.normalize(text_embed, p=2, dim=-1, eps=1e-6)
                # Token-level cross-attn with temperature (sharper for few-shot)
                temperature = getattr(self, 'dcpa_attn_temperature', 0.07)
                attn = torch.matmul(text_for_attn, proto_feat_n.t()) / temperature
                # Class mask: token only attends to its class prototype(s), no cross-class leakage
                valid_indices = torch.where(valid)[0]
                token_class_mask = self._build_token_class_mask(
                    batch_data_samples, text_embed.size(1), valid_indices, text_embed.device)
                if token_class_mask is not None:
                    attn = attn.masked_fill(token_class_mask == 0, -1e9)
                attn = F.softmax(attn, dim=-1)
                delta_text = torch.matmul(attn, proto_value)  # (B, T, C)
                alpha = torch.sigmoid(self.dcpa_alpha_raw).to(
                    device=text_embed.device, dtype=text_embed.dtype)
                text_embed = text_embed + alpha * delta_text
                text_dict['embedded'] = text_embed

        encoder_outputs_dict = self.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict,
            batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        # For L_align = 1 - cos(t', t)
        head_inputs_dict['text_embed_original'] = text_embed_original
        head_inputs_dict['text_embed_adapted'] = text_dict['embedded']
        return head_inputs_dict

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor, valid_ratios: Tensor,
                        text_dict: Dict) -> Dict:
        text_token_mask = text_dict['text_token_mask']
        memory, memory_text = self.encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # for text encoder
            memory_text=text_dict['embedded'],
            text_attention_mask=~text_token_mask,
            position_ids=text_dict['position_ids'],
            text_self_attention_masks=text_dict['masks'])
        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            memory_text=memory_text,
            text_token_mask=text_token_mask)
        return encoder_outputs_dict

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        bs, _, c = memory.shape

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)

        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory, memory_text,
                                     text_token_mask)
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].max_text_len
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        cls_score = enc_outputs_class.max(-1)[0]  # (bs, N)
        score = cls_score
        # Prototype-guided query ranking: rerank within cls top-M for stability.
        # Also reuse proto assignment for query adaptation (below).
        counts = getattr(self, 'prototype_update_counts', None)
        # Fix A3: Require each class has at least pgta_min_updates before PGTA activates.
        pgta_min_pc = int(getattr(self, 'pgta_min_updates', 3))
        if counts is not None:
            _active_counts_pgta = counts[counts > 0]
            _has_proto = _active_counts_pgta.numel() > 0 and int(_active_counts_pgta.min().item()) >= pgta_min_pc
        else:
            _has_proto = False
        proto_unit = None  # (K, embed_dim) normalized, invalid classes zeroed
        valid_proto_mask = None  # (K,) bool
        proto_reliability_all = None  # (K,)
        proto_sim = None  # (bs, N)
        proto_ids = None  # (bs, N)
        if (hasattr(self, 'prototype_memory') and self.prototype_memory is not None
                and _has_proto):
            proto_mem = self.prototype_memory.clone()
            # Fix A2: Per-class valid mask — don't require ALL classes valid (no min()).
            proto_norms = proto_mem.norm(p=2, dim=-1)  # (K,)
            valid_proto_mask = proto_norms >= 1e-6  # (K,) per-class bool
            if valid_proto_mask.any():
                proto_proj_all = self.prototype_proj(proto_mem)  # (K, embed_dim)
                proto_proj_all = proto_proj_all.clone()
                proto_proj_all[~valid_proto_mask] = 0.0  # silence invalid classes
                proto_unit = F.normalize(proto_proj_all, p=2, dim=-1, eps=1e-6)
                proto_unit = proto_unit.clone()
                proto_unit[~valid_proto_mask] = 0.0  # re-zero after normalization
                memory_feat = F.normalize(output_memory, p=2, dim=-1, eps=1e-6)
                proto_sim_all = torch.matmul(memory_feat, proto_unit.t())  # (bs, N, K)
                # Mask invalid class scores to -1 so they don't win argmax.
                proto_sim_all[..., ~valid_proto_mask] = -1.0
                proto_sim, proto_ids = proto_sim_all.max(dim=-1)  # (bs, N)
                proto_reliability_all = torch.clamp(
                    counts.to(device=proto_ids.device, dtype=proto_sim.dtype) /
                    float(getattr(self, 'proto_reliability_threshold', 20.0)),
                    max=1.0)
                # Fix B: Prototype-guided proposal selection.
                # Blend cls_score with prototype similarity weighted by reliability.
                cls_prob = cls_score.sigmoid()
                reliability_per_proposal = proto_reliability_all[proto_ids]  # (bs, N)
                proto_boost = (proto_sim + 1.0) * 0.5 * reliability_per_proposal  # [0,1]
                proto_lambda = float(getattr(self, 'proto_lambda', 0.3))
                blended_score = cls_prob + proto_lambda * proto_boost
                topk_indices = torch.topk(blended_score, k=self.num_queries, dim=1)[1]
            else:
                proto_unit = None
        if proto_unit is None:
            topk_indices = torch.topk(score, k=self.num_queries, dim=1)[1]

        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        # Prototype query adapter (proposal-guided):
        # Use the proto assignment of selected top-k encoder proposals to adapt queries,
        # instead of letting each query slot pick an arbitrary prototype.
        if (proto_unit is not None and proto_reliability_all is not None
                and proto_sim is not None and proto_ids is not None):
            sel_proto_ids = torch.gather(proto_ids, 1, topk_indices)  # (bs, num_queries)
            sel_sim = torch.gather(proto_sim, 1, topk_indices)  # (bs, num_queries) in [-1,1]
            sel_rel = proto_reliability_all[sel_proto_ids]  # (bs, num_queries)
            sel_proto_feat = proto_unit[sel_proto_ids]  # (bs, num_queries, C)
            sel_term = (sel_sim + 1.0) * 0.5  # [0,1]
            # prototype → MLP adapter → reliability gate → query update
            proto_feat = self.proto_query_adapter(sel_proto_feat)
            gate = (sel_rel * sel_term).unsqueeze(-1)
            adapter_scale = float(getattr(self, 'proto_query_adapter_scale', 0.2))
            query = query + adapter_scale * gate * proto_feat
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
        )
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        # append text_feats to head_inputs_dict
        head_inputs_dict['memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask
        return decoder_inputs_dict, head_inputs_dict

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        # from gensim.models import KeyedVectors
        # 目标类别列表
        # text_tuple = ["apple", "avocado", "capsicum", "mango", "orange", "rockmelon", "strawberry"]
        # 加载预训练的词向量
        # word_vectors = KeyedVectors.load_word2vec_format("/home/panjiancheng/projects/NTIRE2025_CDFSOD/weights/GoogleNews-vectors-negative300.bin.gz", binary=True)

        # similar_classes = {
        #     "apple": ['apples', 'peach', 'plum', 'nectarine'],
        #     "avocado": ['avocados', 'pear', 'eggplant'],
        #     "capsicum": ['capsicums', 'chili', 'tomato'],
        #     "mango": ['mangoes', 'mangos', 'papaya', 'persimmon'],
        #     "orange": ['oranges', 'tangerine', 'grapefruit'],
        #     "rockmelon": ['rockmelons', 'cantaloupe', 'honeydew'],
        #     "strawberry": ['strawberries', 'raspberry', 'mulberry', 'berry'],
        # }
        similar_classes = {
            # "car": ['capsicums', 'chili', 'tomato'],
        }

        # # 找到相似类别并用 "or" 连接
        # new_texts = []
        # top_k = 2  # 选择最近的3个相似类别
        # for text in text_prompts_:
        #     if text in word_vectors:
        #         similar_words = [w for w, _ in word_vectors.most_similar(text, topn=top_k)]  # 获取相似类别
        #         new_text = f"{text} or " + " or ".join(similar_words)  # 拼接文本
        #     else:
        #         new_text = text  # 如果词不在 Word2Vec 词汇表，保持原样
        #     new_texts.append(new_text)
        # text_prompts = new_texts
        
        text_prompts = []
        gt_labels = []
        # 生成新的类别描述：训练时保留当前图像出现的类别，并适当采样部分其他类别作为负类，
        # 避免 few-shot 时出现过多负类噪声，同时保持一定的类别判别训练信号。
        max_prompt_classes = getattr(self, 'max_prompt_classes', 20)
        for data_samples in batch_data_samples:
            text_tuple = data_samples.text
            gt_label = data_samples.gt_instances.labels
            dataset_name = self._get_dataset_name(data_samples)
            attrs = self._get_prompt_attrs(dataset_name)

            if self.training and len(gt_label) > 0:
                unique_labels = sorted(set(gt_label.tolist()))
                # 选择一些未出现的类别作为负类（采样式），使训练信号更稳定
                all_indices = list(range(len(text_tuple)))
                neg_candidates = [i for i in all_indices if i not in unique_labels]
                num_neg = max(0, max_prompt_classes - len(unique_labels))
                if len(neg_candidates) > 0 and num_neg > 0:
                    neg_sample = random.sample(neg_candidates, min(len(neg_candidates), num_neg))
                else:
                    neg_sample = []
                selected_indices = sorted(unique_labels + neg_sample)
                text_prompt = tuple(
                    self._make_prompt(text_tuple[i], dataset_name=dataset_name, attrs=attrs)
                    for i in selected_indices)

                # remap gt_label indices to new prompt order
                idx_map = {orig: new_i for new_i, orig in enumerate(selected_indices)}
                gt_label = torch.tensor([idx_map[int(l)] for l in gt_label],
                                        device=gt_label.device)
            else:
                text_prompt = tuple(
                    self._make_prompt(t, dataset_name=dataset_name, attrs=attrs)
                    for t in text_tuple)

            # 加入相似类别扩展（如有配置）
            new_text_tuple = tuple(
                f"{t} or " + " or ".join(similar_classes[t])
                if t in similar_classes else t
                for t in text_prompt)
            text_prompts.append(new_text_tuple)
            gt_labels.append(gt_label)

        new_text_prompts = []
        positive_maps = []
        if len(set(text_prompts)) == 1:
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            tokenized, caption_string, tokens_positive, _ = \
                self.get_tokens_and_prompts(text_prompts[0], True)
            new_text_prompts = [caption_string] * len(batch_inputs)
            for gt_label in gt_labels:
                new_tokens_positive = [tokens_positive[label] for label in gt_label]
                _, positive_map = self.get_positive_map(tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
        else:
            for text_prompt, gt_label in zip(text_prompts, gt_labels):
                tokenized, caption_string, tokens_positive, _ = \
                    self.get_tokens_and_prompts(text_prompt, True)
                new_tokens_positive = [tokens_positive[label] for label in gt_label]
                _, positive_map = self.get_positive_map(tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
                new_text_prompts.append(caption_string)

        text_dict = self.language_model(new_text_prompts)
        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])

        for i, data_samples in enumerate(batch_data_samples):
            positive_map = positive_maps[i].to(
                batch_inputs.device).bool().float()
            text_token_mask = text_dict['text_token_mask'][i]
            data_samples.gt_instances.positive_maps = positive_map
            data_samples.gt_instances.text_token_mask = \
                text_token_mask.unsqueeze(0).repeat(
                    len(positive_map), 1)
        if self.use_autocast:
            with autocast(enabled=True):
                visual_features = self.extract_feat(batch_inputs)
        else:
            visual_features = self.extract_feat(batch_inputs)

        loss_proto = None
        # EMA update prototype_memory from instance (ROI) backbone features (P4 for stability).
        # Optional: update from encoder/decoder features so prototype lives in classification space.
        if self.training and batch_data_samples:
            feat = visual_features[-2] if len(visual_features) > 1 else visual_features[-1]  # (B, C, Hf, Wf)
            B, C, Hf, Wf = feat.shape
            img_h, img_w = batch_inputs.shape[2], batch_inputs.shape[3]
            # Explicit float ratio for roi_align (feature stride may differ from 32).
            spatial_scale = float(Hf) / float(img_h)
            bbox_list = []
            labels_list = []
            for ds in batch_data_samples:
                bboxes = get_box_tensor(ds.gt_instances.bboxes).to(
                    device=feat.device, dtype=feat.dtype)
                if bboxes.shape[0] == 0:
                    bbox_list.append(bboxes)
                    labels_list.append(ds.gt_instances.labels)
                    continue
                if bboxes.max() <= 1.5:
                    bboxes = bboxes * bboxes.new_tensor(
                        [img_w, img_h, img_w, img_h])
                bbox_list.append(bboxes)
                labels_list.append(ds.gt_instances.labels)
            try:
                rois = bbox2roi(bbox_list)
                rois = rois.to(device=feat.device, dtype=feat.dtype)
            except Exception:
                rois = None
            if rois is not None and rois.shape[0] > 0:
                # Per-image global vector: stabilizes prototype when ROI is tiny
                # (small boxes yield noisy roi_align); no dataset-specific branch.
                global_vec = torch.nn.functional.adaptive_avg_pool2d(
                    feat, 1).flatten(1)  # (B, C)
                roi_feats = roi_align(
                    feat, rois, (7, 7), spatial_scale, 0, 'avg', True)
                instance_feats = torch.nn.functional.adaptive_avg_pool2d(
                    roi_feats, 1).flatten(1)  # (N_roi, C)
                labels_flat = torch.cat([
                    lb.to(device=feat.device) for lb in labels_list])
                # rois: [batch_idx, x1, y1, x2, y2] in pixel space
                if rois.shape[1] >= 5 and instance_feats.shape[0] > 0:
                    widths = (rois[:, 3] - rois[:, 1]).abs().clamp(min=1.0)
                    heights = (rois[:, 4] - rois[:, 2]).abs().clamp(min=1.0)
                    areas = widths * heights
                    img_area = max(float(img_h * img_w), 1.0)
                    area_ratio = (areas / img_area).clamp(0.0, 1.0)
                    batch_idx = rois[:, 0].long().clamp(min=0, max=B - 1)
                    for j in range(instance_feats.shape[0]):
                        r = float(area_ratio[j].item())
                        # Tiny box -> blend toward global to reduce noise
                        if r < 0.02:
                            w = min(0.7, (0.02 - r) / 0.02 * 0.7)
                            g = global_vec[batch_idx[j]]
                            instance_feats[j] = (
                                (1.0 - w) * instance_feats[j] + w * g)
                # Prototype contrastive loss: InfoNCE when >=2 valid classes, else cosine fallback
                labels_flat_clip = labels_flat[:instance_feats.shape[0]].long()
                roi_n = F.normalize(instance_feats, p=2, dim=1)
                # Build valid-prototype set (non-zero entries)
                proto_all = self.prototype_memory.to(
                    device=roi_n.device, dtype=roi_n.dtype)
                proto_norms = proto_all.norm(p=2, dim=-1)
                proto_valid_mask = proto_norms >= 1e-6  # (K,)
                K_valid = int(proto_valid_mask.sum().item())
                if K_valid >= 2:
                    # InfoNCE: each ROI should be closest to its own class prototype
                    valid_cls_ids = torch.where(proto_valid_mask)[0]  # (K_v,)
                    proto_valid = F.normalize(
                        proto_all[valid_cls_ids], p=2, dim=-1)  # (K_v, D)
                    cls_to_valid = {
                        int(c): i for i, c in enumerate(valid_cls_ids.tolist())}
                    roi_keep = torch.tensor(
                        [int(l.item()) in cls_to_valid for l in labels_flat_clip],
                        device=roi_n.device)
                    if roi_keep.sum() >= 1:
                        roi_n_c = roi_n[roi_keep]
                        target = torch.tensor(
                            [cls_to_valid[int(l.item())]
                             for l in labels_flat_clip[roi_keep]],
                            device=roi_n.device, dtype=torch.long)
                        logits = (roi_n_c @ proto_valid.t()) / 0.07  # (N', K_v)
                        loss_proto = F.cross_entropy(logits, target)
                elif K_valid == 1 or proto_norms[labels_flat_clip].min().item() >= 1e-6:
                    # Fallback: cosine + symmetric center alignment
                    proto_feat = proto_all[labels_flat_clip]
                    if proto_feat.norm(p=2, dim=1).min().item() >= 1e-6:
                        proto_n = F.normalize(proto_feat, p=2, dim=1)
                        cos_sim = (roi_n * proto_n).sum(dim=1)
                        loss_proto = (1 - cos_sim).mean()
                        proto_center = proto_n.mean(dim=0, keepdim=True)
                        roi_center = roi_n.mean(dim=0, keepdim=True)
                        loss_proto_sym = (
                            1 - F.cosine_similarity(proto_center, roi_center, dim=1)
                        ).mean()
                        loss_proto = (loss_proto + loss_proto_sym) * 0.5
                with torch.no_grad():
                    labels_flat = labels_flat[:instance_feats.shape[0]]
                    for class_id_t in labels_flat.unique():
                        l_val = int(class_id_t.long().item())
                        if l_val < 0 or l_val >= self.prototype_memory.shape[0]:
                            continue
                        mask = labels_flat == class_id_t
                        if mask.sum() == 0:
                            continue
                        # Similarity filtering: only update from ROIs similar to current prototype (reduce noise).
                        proto_old = self.prototype_memory.data[l_val].to(
                            device=instance_feats.device, dtype=instance_feats.dtype)
                        if proto_old.norm(p=2).item() >= 1e-6:
                            # Similarity filtering is skipped: projection weights change each epoch,
                            # making stored prototype mis-aligned with current projection.
                            # With few-shot (1 clean instance per class), no noise to filter.
                            class_feat = instance_feats[mask].mean(dim=0)
                        else:
                            class_feat = instance_feats[mask].mean(dim=0)
                        if hasattr(self, 'proto_feat_proj'):
                            class_feat = self.proto_feat_proj(class_feat)
                        class_feat = F.normalize(class_feat, p=2, dim=0)
                        n_inst = int(mask.sum().item())
                        prev_updates = int(self.prototype_update_counts[l_val].item())
                        if prev_updates == 0:
                            # First update: direct assignment, no EMA dilution
                            alpha_ema = 1.0
                        else:
                            alpha_ema = 0.1 * min(1.0, n_inst / 4.0)
                            maturity = max(0.0, min(1.0, (prev_updates - 2) / 8.0))
                            alpha_ema = alpha_ema * (1.0 + 1.2 * maturity)
                            alpha_ema = max(0.02, min(0.20, alpha_ema))
                            if prev_updates >= int(self.pgta_min_updates):
                                alpha_ema = max(alpha_ema, 0.06)
                        beta_ema = 1.0 - alpha_ema
                        self.prototype_memory.data[l_val] = (
                            beta_ema * self.prototype_memory.data[l_val]
                            + alpha_ema * class_feat.detach().to(
                                self.prototype_memory.device))
                        self.prototype_update_counts[l_val] += 1
            else:
                global_feat = torch.nn.functional.adaptive_avg_pool2d(
                    feat, 1).flatten(1)
                with torch.no_grad():
                    for i, data_sample in enumerate(batch_data_samples):
                        labels = data_sample.gt_instances.labels
                        for l in labels.unique():
                            l_val = l.item()
                            self.prototype_memory.data[l_val] = (
                                0.9 * self.prototype_memory.data[l_val]
                                + 0.1 * global_feat[i].detach().to(
                                    self.prototype_memory.device))
                            self.prototype_update_counts[l_val] += 1

        head_inputs_dict = self.forward_transformer(
            visual_features, text_dict, batch_data_samples)

        # DCPA: semantic alignment loss L_align = 1 - cos(t', t)
        text_embed_original = head_inputs_dict.pop('text_embed_original', None)
        text_embed_adapted = head_inputs_dict.pop('text_embed_adapted', None)

        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)
        if loss_proto is not None:
            losses['loss_proto'] = loss_proto * getattr(
                self, 'loss_proto_weight', 0.1)
        if (self.training and text_embed_original is not None
                and text_embed_adapted is not None
                and hasattr(self, 'loss_align_weight')):
            cos_sim = F.cosine_similarity(
                text_embed_adapted.flatten(0, 1),
                text_embed_original.flatten(0, 1),
                dim=-1)
            loss_align = (1 - cos_sim).mean()
            losses['loss_align'] = loss_align * self.loss_align_weight
        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        text_prompts = []
        enhanced_text_prompts = []
        tokens_positives = []
        for data_samples in batch_data_samples:
            # If using custom_entities, assume text is already formatted as desired.
            if getattr(data_samples, 'custom_entities', False):
                text_prompts.append(data_samples.text)
            else:
                dataset_name = self._get_dataset_name(data_samples)
                attrs = self._get_prompt_attrs(dataset_name)
                text_prompts.append(
                    tuple(self._make_prompt(t, dataset_name=dataset_name, attrs=attrs)
                          for t in data_samples.text))

            if 'caption_prompt' in data_samples:
                enhanced_text_prompts.append(data_samples.caption_prompt)
            else:
                enhanced_text_prompts.append(None)
            tokens_positives.append(data_samples.get('tokens_positive', None))

        if 'custom_entities' in batch_data_samples[0]:
            # Assuming that the `custom_entities` flag
            # inside a batch is always the same. For single image inference
            custom_entities = batch_data_samples[0].custom_entities
        else:
            custom_entities = False
        if len(text_prompts) == 1:
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    text_prompts[0], custom_entities, enhanced_text_prompts[0],
                    tokens_positives[0])
            ] * len(batch_inputs)
        else:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(text_prompt,
                                                     custom_entities,
                                                     enhanced_text_prompt,
                                                     tokens_positive)
                for text_prompt, enhanced_text_prompt, tokens_positive in zip(
                    text_prompts, enhanced_text_prompts, tokens_positives)
            ]
        token_positive_maps, text_prompts, _, entities = zip(
            *_positive_maps_and_prompts)

        # image feature extraction
        visual_feats = self.extract_feat(batch_inputs)

        if isinstance(text_prompts[0], list):
            # chunked text prompts, only bs=1 is supported
            assert len(batch_inputs) == 1
            count = 0
            results_list = []

            entities = [[item for lst in entities[0] for item in lst]]

            for b in range(len(text_prompts[0])):
                text_prompts_once = [text_prompts[0][b]]
                token_positive_maps_once = token_positive_maps[0][b]
                text_dict = self.language_model(text_prompts_once)
                # text feature map layer
                if self.text_feat_map is not None:
                    text_dict['embedded'] = self.text_feat_map(
                        text_dict['embedded'])

                batch_data_samples[
                    0].token_positive_map = token_positive_maps_once

                head_inputs_dict = self.forward_transformer(
                    copy.deepcopy(visual_feats), text_dict, batch_data_samples)
                head_inputs_dict.pop('text_embed_original', None)
                head_inputs_dict.pop('text_embed_adapted', None)
                pred_instances = self.bbox_head.predict(
                    **head_inputs_dict,
                    rescale=rescale,
                    batch_data_samples=batch_data_samples)[0]

                if len(pred_instances) > 0:
                    pred_instances.labels += count
                count += len(token_positive_maps_once)
                results_list.append(pred_instances)
            results_list = [results_list[0].cat(results_list)]
            is_rec_tasks = [False] * len(results_list)
        else:
            # extract text feats
            text_dict = self.language_model(list(text_prompts))
            # text feature map layer
            if self.text_feat_map is not None:
                text_dict['embedded'] = self.text_feat_map(
                    text_dict['embedded'])

            is_rec_tasks = []
            for i, data_samples in enumerate(batch_data_samples):
                if token_positive_maps[i] is not None:
                    is_rec_tasks.append(False)
                else:
                    is_rec_tasks.append(True)
                data_samples.token_positive_map = token_positive_maps[i]

            head_inputs_dict = self.forward_transformer(
                visual_feats, text_dict, batch_data_samples)
            head_inputs_dict.pop('text_embed_original', None)
            head_inputs_dict.pop('text_embed_adapted', None)
            results_list = self.bbox_head.predict(
                **head_inputs_dict,
                rescale=rescale,
                batch_data_samples=batch_data_samples)

        for data_sample, pred_instances, entity, is_rec_task in zip(
                batch_data_samples, results_list, entities, is_rec_tasks):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if is_rec_task:
                        label_names.append(entity)
                        continue
                    if labels >= len(entity):
                        warnings.warn(
                            'The unexpected output indicates an issue with '
                            'named entity recognition. You can try '
                            'setting custom_entities=True and running '
                            'again to see if it helps.')
                        label_names.append('unobject')
                    else:
                        label_names.append(entity[labels])
                # for visualization
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances
        return batch_data_samples
