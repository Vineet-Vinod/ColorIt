"""Compiled, reusable Qwen Image Edit denoising for fixed-shape batches."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Literal


CFGMode = Literal["auto", "fused", "separate"]


def qwen_2511_transformer_call(
    transformer,
    timestep,
    config,
    hidden_states,
    prompt_embeds,
    prompt_mask,
    cond_image_grid,
):
    """Run the 2511 transformer with a dynamic MLX timestep scalar."""

    from .qwen_2511 import zero_cond_transformer_forward

    return zero_cond_transformer_forward(
        transformer,
        timestep,
        config,
        hidden_states,
        prompt_embeds,
        prompt_mask,
        cond_image_grid,
    )


def qwen_transformer_call(
    transformer,
    timestep,
    config,
    hidden_states,
    prompt_embeds,
    prompt_mask,
    cond_image_grid,
):
    """Run either Qwen 2511 or FireRed's standard timestep conditioning."""

    from .qwen_2511 import qwen_transformer_forward

    return qwen_transformer_forward(
        transformer,
        timestep,
        config,
        hidden_states,
        prompt_embeds,
        prompt_mask,
        cond_image_grid,
    )


@dataclass
class CompileStats:
    """Small observable counters for tests and benchmark logs."""

    positive_graph_builds: int = 0
    negative_graph_builds: int = 0
    fused_graph_builds: int = 0
    denoise_steps: int = 0


class CompiledQwenCFG:
    """Keep fixed-shape Qwen conditioning and compiled CFG graphs resident.

    The instance is valid only for one image shape and one batch size. It can
    run any number of denoise steps and seed batches without re-encoding the
    prompt or source image. ``auto`` fuses CFG for one seed at a time. Larger
    batches use separate positive and negative passes to avoid doubling the
    dominant activation allocation.
    """

    def __init__(
        self,
        *,
        transformer,
        config,
        static_image_latents,
        positive_prompt_embeds,
        positive_prompt_mask,
        negative_prompt_embeds,
        negative_prompt_mask,
        cond_image_grid,
        guidance: float,
        cfg_mode: CFGMode = "auto",
        fused_cfg_max_batch: int = 1,
        transformer_call: Callable = qwen_transformer_call,
    ) -> None:
        if cfg_mode not in {"auto", "fused", "separate"}:
            raise ValueError("cfg_mode must be auto, fused, or separate")
        if fused_cfg_max_batch < 1:
            raise ValueError("fused_cfg_max_batch must be positive")
        self.transformer = transformer
        self.config = config
        self.static_image_latents = static_image_latents
        (
            self.positive_prompt_embeds,
            self.positive_prompt_mask,
            self.negative_prompt_embeds,
            self.negative_prompt_mask,
        ) = self._pad_prompt_pair(
            positive_prompt_embeds,
            positive_prompt_mask,
            negative_prompt_embeds,
            negative_prompt_mask,
        )
        self.cond_image_grid = cond_image_grid
        self.guidance = guidance
        self.cfg_mode = cfg_mode
        self.fused_cfg_max_batch = fused_cfg_max_batch
        self.transformer_call = transformer_call
        self.stats = CompileStats()
        self._positive_predict = None
        self._negative_predict = None
        self._fused_predict = None

    @staticmethod
    def _pad_prompt_pair(positive, positive_mask, negative, negative_mask):
        """Pad CFG text sequences so they can share one transformer batch."""

        import mlx.core as mx

        token_count = max(positive.shape[1], negative.shape[1])

        def pad(embeds, mask):
            missing = token_count - embeds.shape[1]
            if not missing:
                return embeds, mask
            embeds = mx.pad(embeds, ((0, 0), (0, missing), (0, 0)))
            mask = mx.pad(mask, ((0, 0), (0, missing)))
            return embeds, mask

        positive, positive_mask = pad(positive, positive_mask)
        negative, negative_mask = pad(negative, negative_mask)
        return positive, positive_mask, negative, negative_mask

    @staticmethod
    def _guided_noise(positive, negative, guidance):
        import mlx.core as mx

        combined = negative + guidance * (positive - negative)
        positive_norm = mx.sqrt(mx.sum(positive * positive, axis=-1, keepdims=True) + 1e-12)
        combined_norm = mx.sqrt(mx.sum(combined * combined, axis=-1, keepdims=True) + 1e-12)
        return combined * (positive_norm / combined_norm)

    def _build_separate(self) -> None:
        if self._positive_predict is not None:
            return
        import mlx.core as mx

        def positive(latents, timestep):
            self.stats.positive_graph_builds += 1
            hidden_states = mx.concatenate([latents, self.static_image_latents], axis=1)
            return self.transformer_call(
                self.transformer,
                timestep,
                self.config,
                hidden_states,
                self.positive_prompt_embeds,
                self.positive_prompt_mask,
                self.cond_image_grid,
            )[:, : latents.shape[1]]

        def negative(latents, timestep):
            self.stats.negative_graph_builds += 1
            hidden_states = mx.concatenate([latents, self.static_image_latents], axis=1)
            return self.transformer_call(
                self.transformer,
                timestep,
                self.config,
                hidden_states,
                self.negative_prompt_embeds,
                self.negative_prompt_mask,
                self.cond_image_grid,
            )[:, : latents.shape[1]]

        self._positive_predict = mx.compile(positive)
        self._negative_predict = mx.compile(negative)

    def _build_fused(self) -> None:
        if self._fused_predict is not None:
            return
        import mlx.core as mx

        def fused(latents, timestep):
            self.stats.fused_graph_builds += 1
            repeated_latents = mx.concatenate([latents, latents], axis=0)
            repeated_static = mx.concatenate([self.static_image_latents, self.static_image_latents], axis=0)
            hidden_states = mx.concatenate([repeated_latents, repeated_static], axis=1)
            prompt_embeds = mx.concatenate([self.positive_prompt_embeds, self.negative_prompt_embeds], axis=0)
            prompt_mask = mx.concatenate([self.positive_prompt_mask, self.negative_prompt_mask], axis=0)
            return self.transformer_call(
                self.transformer,
                timestep,
                self.config,
                hidden_states,
                prompt_embeds,
                prompt_mask,
                self.cond_image_grid,
            )[:, : latents.shape[1]]

        self._fused_predict = mx.compile(fused)

    def _can_fuse(self, batch_size: int) -> bool:
        return self.cfg_mode == "fused" or (
            self.cfg_mode == "auto" and batch_size <= self.fused_cfg_max_batch
        )

    def predict_noise(self, latents, timestep):
        """Run compiled CFG once. ``timestep`` is an MLX scalar, not a Python int."""

        batch_size = latents.shape[0]
        if batch_size != self.static_image_latents.shape[0]:
            raise ValueError("latents and prepared conditioning must have the same batch size")
        if self._can_fuse(batch_size):
            self._build_fused()
            noise = self._fused_predict(latents, timestep)
            positive, negative = noise[:batch_size], noise[batch_size:]
        else:
            self._build_separate()
            positive = self._positive_predict(latents, timestep)
            negative = self._negative_predict(latents, timestep)
        self.stats.denoise_steps += 1
        return self._guided_noise(positive, negative, self.guidance)


def benchmark_fixed_shape(cfg: CompiledQwenCFG, latents, timesteps) -> dict[str, float | int]:
    """Measure fixed-shape compiled CFG execution without loading Qwen weights."""

    import mlx.core as mx

    started = perf_counter()
    for timestep in timesteps:
        latents = cfg.predict_noise(latents, timestep)
    mx.eval(latents)
    elapsed = perf_counter() - started
    return {
        "seconds": elapsed,
        "steps": cfg.stats.denoise_steps,
        "positive_graph_builds": cfg.stats.positive_graph_builds,
        "negative_graph_builds": cfg.stats.negative_graph_builds,
        "fused_graph_builds": cfg.stats.fused_graph_builds,
    }


@dataclass
class PreparedQwenEdit:
    """A prompt and reference image encoded once for a fixed seed batch."""

    cfg: CompiledQwenCFG
    config: object
    batch_size: int


class QwenCompiledEditLoop:
    """Prepare and denoise Qwen Image Edit batches without reloading work."""

    def __init__(self, model, *, cfg_mode: CFGMode = "auto", fused_cfg_max_batch: int = 1) -> None:
        self.model = model
        self.cfg_mode = cfg_mode
        self.fused_cfg_max_batch = fused_cfg_max_batch

    @staticmethod
    def _expand_batch(value, batch_size: int):
        import mlx.core as mx

        if value.shape[0] == batch_size:
            return value
        if value.shape[0] != 1:
            raise ValueError("prepared Qwen condition must have batch size one or the requested batch size")
        return mx.broadcast_to(value, (batch_size, *value.shape[1:]))

    def prepare(
        self,
        *,
        image_path: str,
        prompt: str,
        negative_prompt: str,
        width: int | None,
        height: int | None,
        steps: int,
        guidance: float,
        scheduler: str,
        batch_size: int,
    ) -> PreparedQwenEdit:
        """Encode source image and prompts once for a same-shape seed batch."""

        from mflux.models.qwen.variants.edit.qwen_edit_util import QwenEditUtil

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        config, vl_width, vl_height, vae_width, vae_height = self.model._compute_dimensions(
            width=width,
            height=height,
            guidance=guidance,
            scheduler=scheduler,
            image_path=image_path,
            image_paths=[image_path],
            num_inference_steps=steps,
        )
        positive, positive_mask, negative, negative_mask = self.model._encode_prompts_with_images(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_paths=[image_path],
            config=config,
            vl_width=vl_width,
            vl_height=vl_height,
        )
        static, _, cond_height, cond_width, number_images = QwenEditUtil.create_image_conditioning_latents(
            vae=self.model.vae,
            width=vae_width,
            height=vae_height,
            image_paths=[image_path],
            tiling_config=self.model.tiling_config,
        )
        if number_images != 1:
            raise ValueError("Qwen keyframe colorization requires one source image per prepared batch")
        cond_grid = (1, cond_height, cond_width)
        cfg = CompiledQwenCFG(
            transformer=self.model.transformer,
            config=config,
            static_image_latents=self._expand_batch(static, batch_size),
            positive_prompt_embeds=self._expand_batch(positive, batch_size),
            positive_prompt_mask=self._expand_batch(positive_mask, batch_size),
            negative_prompt_embeds=self._expand_batch(negative, batch_size),
            negative_prompt_mask=self._expand_batch(negative_mask, batch_size),
            cond_image_grid=cond_grid,
            guidance=guidance,
            cfg_mode=self.cfg_mode,
            fused_cfg_max_batch=self.fused_cfg_max_batch,
        )
        return PreparedQwenEdit(cfg=cfg, config=config, batch_size=batch_size)

    def denoise(self, prepared: PreparedQwenEdit, seeds: list[int]):
        """Denoise all prepared seeds, retaining its compiled graphs across steps."""

        import mlx.core as mx
        from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator

        if len(seeds) != prepared.batch_size:
            raise ValueError("seed count must match the prepared batch size")
        latents = mx.concatenate(
            [QwenLatentCreator.create_noise(seed, prepared.config.height, prepared.config.width) for seed in seeds],
            axis=0,
        )
        return self.denoise_latents(prepared, latents)

    def denoise_latents(self, prepared: PreparedQwenEdit, latents):
        """Denoise supplied fixed-shape latents. Useful for parity tests."""

        import mlx.core as mx

        if latents.shape[0] != prepared.batch_size:
            raise ValueError("latent batch size must match prepared conditioning")
        scheduler = prepared.config.scheduler
        for index in range(prepared.config.num_inference_steps):
            # MFLUX's eager transformer receives the integer loop index and
            # converts it to scheduler.sigmas[index]. Passing the public
            # integer-valued `timesteps` array here over-scales conditioning
            # by roughly 1000x and decodes as colored noise.
            timestep = scheduler.sigmas[index]
            noise = prepared.cfg.predict_noise(latents, timestep)
            latents = scheduler.step(noise=noise, timestep=index, latents=latents)
            # MLX is lazy. Materialize every scheduler state before the next
            # transformer call instead of retaining a full multi-step graph
            # and all of its activations until VAE decode.
            mx.eval(latents)
        return latents

    @staticmethod
    def _unpack_batch(latents, height: int, width: int):
        """Unpack Qwen latents without discarding a batch dimension."""

        import mlx.core as mx

        batch_size = latents.shape[0]
        latent_height = height // 16
        latent_width = width // 16
        latents = mx.reshape(latents, (batch_size, latent_height, latent_width, 16, 2, 2))
        latents = mx.transpose(latents, (0, 3, 1, 4, 2, 5))
        return mx.reshape(latents, (batch_size, 16, latent_height * 2, latent_width * 2))

    def generate_batch(
        self,
        *,
        image_path: str,
        prompt: str,
        negative_prompt: str,
        seeds: list[int],
        width: int | None,
        height: int | None,
        steps: int,
        guidance: float,
        scheduler: str,
    ) -> list[object]:
        """Generate and decode all seeds using one encoded source and prompt."""

        from mflux.models.common.vae.vae_util import VAEUtil
        from mflux.utils.image_util import ImageUtil

        if not seeds:
            raise ValueError("seeds must not be empty")
        prepared = self.prepare(
            image_path=image_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            guidance=guidance,
            scheduler=scheduler,
            batch_size=len(seeds),
        )
        started = perf_counter()
        latents = self.denoise(prepared, seeds)
        decoded = VAEUtil.decode(
            vae=self.model.vae,
            latent=self._unpack_batch(latents, prepared.config.height, prepared.config.width),
            tiling_config=self.model.tiling_config,
        )
        generation_time = perf_counter() - started
        images = []
        for index, seed in enumerate(seeds):
            images.append(
                ImageUtil.to_image(
                    decoded_latents=decoded[index : index + 1],
                    config=prepared.config,
                    seed=seed,
                    prompt=prompt,
                    quantization=self.model.bits,
                    lora_paths=self.model.lora_paths,
                    lora_scales=self.model.lora_scales,
                    image_path=image_path,
                    image_paths=[image_path],
                    generation_time=generation_time / len(seeds),
                    negative_prompt=negative_prompt,
                )
            )
        return images
