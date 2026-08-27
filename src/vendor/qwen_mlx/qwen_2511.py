"""2511's zero-condition-timestep correction for mflux 0.19.1.

Qwen Image Edit 2511 adds ``zero_cond_t`` to the transformer configuration.
It gives source-image tokens a zero timestep while generated tokens keep the
denoising timestep. mflux 0.19.1 predates that flag, so this runtime patch adds
the official diffusers behavior without forking mflux.
"""

from __future__ import annotations


def zero_condition_timesteps(timestep):
    """Pair denoising timesteps with zero timesteps for source-image tokens."""

    import mlx.core as mx

    return mx.concatenate([timestep, mx.zeros_like(timestep)], axis=0)


def token_modulation_index(batch_size: int, generated_tokens: int, source_tokens: int):
    """Return 0 for generated tokens and 1 for source-image conditioning tokens."""

    import mlx.core as mx

    if generated_tokens < 0 or source_tokens < 0:
        raise ValueError("token counts must be non-negative")
    return mx.concatenate(
        [
            mx.zeros((batch_size, generated_tokens), dtype=mx.int32),
            mx.ones((batch_size, source_tokens), dtype=mx.int32),
        ],
        axis=1,
    )


def apply_zero_condition_modulation(x, mod_params, index):
    """Select normal or zero-timestep modulation for each image token.

    ``mod_params`` has one row for generated tokens and one for source-image
    tokens. This matches the official Qwen 2511 ``zero_cond_t`` behavior.
    """

    import mlx.core as mx

    shift, scale, gate = mx.split(mod_params, 3, axis=-1)
    batch_size = shift.shape[0] // 2
    if batch_size * 2 != shift.shape[0]:
        raise ValueError("Qwen 2511 expected paired timestep embeddings")
    index = index[..., None]
    shift = mx.where(index == 0, shift[:batch_size, None, :], shift[batch_size:, None, :])
    scale = mx.where(index == 0, scale[:batch_size, None, :], scale[batch_size:, None, :])
    gate = mx.where(index == 0, gate[:batch_size, None, :], gate[batch_size:, None, :])
    return x * (1 + scale) + shift, gate


def enable_zero_cond_t(model) -> None:
    """Enable Qwen 2511 token-wise timestep conditioning on one mflux model."""

    import mlx.core as mx
    from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer
    from mflux.models.qwen.model.qwen_transformer.qwen_transformer_block import QwenTransformerBlock

    if getattr(QwenTransformer, "_colorit_2511_patch", False):
        model.transformer.zero_cond_t = True
        for block in model.transformer.transformer_blocks:
            block.zero_cond_t = True
        return

    original_transformer_call = QwenTransformer.__call__
    original_block_call = QwenTransformerBlock.__call__
    original_modulate = QwenTransformerBlock._modulate

    def modulate(x, mod_params, index=None):
        if index is None:
            return original_modulate(x, mod_params)
        return apply_zero_condition_modulation(x, mod_params, index)

    def block_call(
        self,
        hidden_states,
        encoder_hidden_states,
        encoder_hidden_states_mask,
        text_embeddings,
        image_rotary_emb,
        block_idx=None,
        modulate_index=None,
    ):
        if not getattr(self, "zero_cond_t", False):
            return original_block_call(
                self,
                hidden_states,
                encoder_hidden_states,
                encoder_hidden_states_mask,
                text_embeddings,
                image_rotary_emb,
                block_idx,
            )
        img_mod_params = self.img_mod_linear(self.img_mod_silu(text_embeddings))
        txt_embeddings = text_embeddings[: text_embeddings.shape[0] // 2]
        txt_mod_params = self.txt_mod_linear(self.txt_mod_silu(txt_embeddings))
        img_mod1, img_mod2 = mx.split(img_mod_params, 2, axis=-1)
        txt_mod1, txt_mod2 = mx.split(txt_mod_params, 2, axis=-1)
        img_modulated, img_gate1 = modulate(self.img_norm1(hidden_states), img_mod1, modulate_index)
        txt_modulated, txt_gate1 = modulate(self.txt_norm1(encoder_hidden_states), txt_mod1)
        img_attn_output, txt_attn_output = self.attn(
            img_modulated=img_modulated,
            txt_modulated=txt_modulated,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            block_idx=block_idx,
        )
        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output
        img_modulated2, img_gate2 = modulate(self.img_norm2(hidden_states), img_mod2, modulate_index)
        hidden_states = hidden_states + img_gate2 * self.img_ff(img_modulated2)
        txt_modulated2, txt_gate2 = modulate(self.txt_norm2(encoder_hidden_states), txt_mod2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * self.txt_ff(txt_modulated2)
        return encoder_hidden_states, hidden_states

    def transformer_call(
        self,
        t,
        config,
        hidden_states,
        encoder_hidden_states,
        encoder_hidden_states_mask,
        qwen_image_ids=None,
        cond_image_grid=None,
    ):
        if not getattr(self, "zero_cond_t", False):
            return original_transformer_call(
                self,
                t,
                config,
                hidden_states,
                encoder_hidden_states,
                encoder_hidden_states_mask,
                qwen_image_ids,
                cond_image_grid,
            )
        hidden_states = self.img_in(hidden_states)
        batch_size = hidden_states.shape[0]
        timestep = self._compute_timestep(t, config)
        timestep = mx.broadcast_to(timestep, (batch_size,)).astype(hidden_states.dtype)
        timestep = zero_condition_timesteps(timestep)
        encoder_hidden_states = self.txt_in(self.txt_norm(encoder_hidden_states))
        text_embeddings = self.time_text_embed(timestep, hidden_states)
        image_rotary_embeddings = self._compute_rotary_embeddings(
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            pos_embed=self.pos_embed,
            config=config,
            cond_image_grid=cond_image_grid,
        )
        conditioned_tokens = 0
        if cond_image_grid is not None:
            grids = cond_image_grid if isinstance(cond_image_grid, list) else [cond_image_grid]
            conditioned_tokens = sum(height * width for _, height, width in grids)
        generated_tokens = hidden_states.shape[1] - conditioned_tokens
        if generated_tokens < 0:
            raise ValueError("Qwen 2511 conditioning grid exceeds image token count")
        modulate_index = token_modulation_index(batch_size, generated_tokens, conditioned_tokens)
        for index, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                text_embeddings=text_embeddings,
                image_rotary_emb=image_rotary_embeddings,
                block_idx=index,
                modulate_index=modulate_index,
            )
        text_embeddings = text_embeddings[: text_embeddings.shape[0] // 2]
        return self.proj_out(self.norm_out(hidden_states, text_embeddings))

    QwenTransformerBlock._modulate = staticmethod(modulate)
    QwenTransformerBlock.__call__ = block_call
    QwenTransformer.__call__ = transformer_call
    QwenTransformer._colorit_2511_patch = True
    model.transformer.zero_cond_t = True
    for block in model.transformer.transformer_blocks:
        block.zero_cond_t = True
