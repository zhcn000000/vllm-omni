from __future__ import annotations

import base64
import random
import time
from http import HTTPStatus
from io import BytesIO
from typing import TYPE_CHECKING, Any, cast

import httpx
from fastapi import HTTPException, Request
from PIL import Image
from vllm import SamplingParams
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.logger import logger

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.image_api_utils import (
    apply_stage_default_sampling_params,
    encode_image_base64,
    parse_size,
)
from vllm_omni.entrypoints.openai.protocol.images import (
    ImageData,
    ImageEditRequest,
    ImageEditResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
)
from vllm_omni.entrypoints.openai.vision_utils_mixin import VisionMixin
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniSamplingParams, OmniTextPrompt

if TYPE_CHECKING:
    from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion


class OmniOpenAIServingImage(VisionMixin):
    """OpenAI-style image generation handler for omni diffusion models."""

    def __init__(
        self,
        engine_client: EngineClient,
        model_name: str | None = None,
        stage_configs: list[Any] | None = None,
    ) -> None:
        self._engine_client = engine_client
        self._model_name = model_name
        self._stage_configs = stage_configs

        self._stage_types = self._resolve_stage_types(stage_configs)

    def _resolve_stage_types(self, stage_configs: list[Any] | None = None) -> list[str]:
        resolved_stage_configs = stage_configs
        if resolved_stage_configs is None:
            resolved_stage_configs = cast(list[Any] | None, getattr(self._engine_client, "stage_configs", None))

        if not resolved_stage_configs:
            return []

        stage_types: list[str] = []
        for stage in resolved_stage_configs:
            stage_types.append(self._get_stage_type(stage))
        return stage_types

    @classmethod
    def for_diffusion(
        cls,
        diffusion_engine: AsyncOmniDiffusion,
        model_name: str | None = None,
        stage_configs: list[Any] | None = None,
    ) -> OmniOpenAIServingImage:
        return cls(
            diffusion_engine,  # type: ignore[arg-type]
            model_name=model_name,
            stage_configs=stage_configs,
        )

    async def _generate_with_async_omni(
        self,
        gen_params: OmniDiffusionSamplingParams,
        **kwargs,
    ):
        engine_client = self.engine_client
        engine_client = cast(AsyncOmni, engine_client)
        result = None
        stage_list = getattr(engine_client, "stage_list", None)
        if isinstance(stage_list, list):
            default_params_list: list[OmniSamplingParams] | None = getattr(
                engine_client, "default_sampling_params_list", None
            )
            if not isinstance(default_params_list, list):
                default_params_list = [
                    OmniDiffusionSamplingParams() if st == "diffusion" else SamplingParams() for st in self._stage_types
                ]
            else:
                default_params_list = list(default_params_list)
            if len(default_params_list) != len(self._stage_types):
                default_params_list = (
                    default_params_list
                    + [
                        OmniDiffusionSamplingParams() if st == "diffusion" else SamplingParams()
                        for st in self._stage_types
                    ]
                )[: len(self._stage_types)]

            sampling_params_list: list[OmniSamplingParams] = []
            for idx, stage_type in enumerate(self._stage_types):
                if stage_type == "diffusion":
                    sampling_params_list.append(gen_params)
                else:
                    base_params = default_params_list[idx]
                    sampling_params_list.append(base_params)

            async for output in engine_client.generate(
                sampling_params_list=sampling_params_list,
                **kwargs,
            ):
                result = output
        else:
            async for output in engine_client.generate(
                sampling_params_list=[gen_params],
                **kwargs,
            ):
                result = output

        if result is None:
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                detail="No output generated from multi-stage pipeline.",
            )
        return result

    @staticmethod
    def _encode_image_base64_with_compression(
        image: Image.Image, format: str = "png", output_compression: int = 100
    ) -> str:
        """Encode PIL Image to base64 PNG string.

        Args:
            image: PIL Image object
            format: Output image format (e.g., "PNG", "JPEG", "WEBP")
            output_compression: Compression level (0-100%), 100 for best quality
        Returns:
            Base64-encoded image as string
        """
        buffer = BytesIO()
        normalized_format = format.lower()
        save_kwargs = {}
        if normalized_format in ("jpg", "jpeg", "webp"):
            save_kwargs["quality"] = output_compression
        elif normalized_format == "png":
            save_kwargs["compress_level"] = max(0, min(9, 9 - output_compression // 11))  # Map 0-100 to 9-0

        image.save(buffer, format=normalized_format, **save_kwargs)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    @staticmethod
    async def _load_input_images(
        inputs: list[Any],
    ) -> list[Image.Image]:
        """
        convert to PIL.Image.Image list
        """
        if isinstance(inputs, str):
            inputs = [inputs]

        images: list[Image.Image] = []

        for inp in inputs:
            # 1. URL + base64
            if isinstance(inp, str) and inp.startswith("data:image"):
                try:
                    _, b64_data = inp.split(",", 1)
                    image_bytes = base64.b64decode(b64_data)
                    img = Image.open(BytesIO(image_bytes))
                    images.append(img)
                except Exception as e:
                    raise ValueError(f"Invalid base64 image: {e}")

            # 2. URL
            elif isinstance(inp, str) and inp.startswith("http"):
                async with httpx.AsyncClient(timeout=60) as client:
                    try:
                        resp = await client.get(inp)
                        resp.raise_for_status()
                        img = Image.open(BytesIO(resp.content))
                        images.append(img)
                    except Exception as e:
                        raise ValueError(f"Failed to download image from URL {inp}: {e}")

            # 3. UploadFile
            elif hasattr(inp, "file"):
                try:
                    upload_file = cast(Any, inp)
                    img_data = await upload_file.read()
                    img = Image.open(BytesIO(img_data))
                    images.append(img)
                except Exception as e:
                    raise ValueError(f"Failed to open uploaded file: {e}")
            else:
                raise ValueError(f"Unsupported input: {inp}")

        if not images:
            raise ValueError("No valid input images found")

        return images

    @staticmethod
    def _choose_output_format(output_format: str | None, background: str | None) -> str:
        # Normalize and choose extension
        fmt = (output_format or "").lower()
        if fmt in ("jpg", "png", "webp", "jpeg"):
            return fmt
        # If transparency requested, prefer png
        if (background or "auto").lower() == "transparent":
            return "png"
        # Default
        return "jpeg"

    async def generate_image(
        self,
        request: ImageGenerationRequest,
        raw_request: Request | None = None,
    ) -> ImageGenerationResponse | ErrorResponse:
        """Process image generation request."""
        engine_client = self.engine_client
        engine_client = cast(AsyncOmni, engine_client)
        model_name = self._resolve_model_name(raw_request)

        if request.model is not None and model_name is not None and request.model != model_name:
            logger.warning(
                "Model mismatch: request specifies '%s' but server is running '%s'. Using server model.",
                request.model,
                model_name,
            )

        prompt = OmniTextPrompt(
            prompt=request.prompt,
        )
        if request.negative_prompt is not None:
            prompt["negative_prompt"] = request.negative_prompt

        gen_params = OmniDiffusionSamplingParams(num_outputs_per_prompt=request.n)

        app_state_args = None
        if raw_request is not None:
            app_state_args = getattr(raw_request.app.state, "args", None)
        default_sample_param = getattr(app_state_args, "default_sampling_params", None)
        fallback_stage_configs = (
            cast(list[Any] | None, getattr(raw_request.app.state, "stage_configs", None))
            if raw_request is not None
            else None
        )
        stage_types = self._stage_types or self._resolve_stage_types(fallback_stage_configs)
        diffusion_stage_ids = [i for i, t in enumerate(stage_types) if t == "diffusion"]
        if not diffusion_stage_ids:
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail="No diffusion stage configured for image generation.",
            )
        apply_stage_default_sampling_params(
            default_sample_param,
            gen_params,
            str(diffusion_stage_ids[0]),
        )

        lora_request, lora_scale = self._parse_lora_request(request.lora)
        if lora_request:
            gen_params.lora_request = lora_request
        if lora_scale is not None:
            gen_params.lora_scale = lora_scale
        width, height = None, None
        if request.size:
            width, height = parse_size(request.size)
            gen_params.width = width
            gen_params.height = height
            size_str = f"{width}x{height}"
        else:
            size_str = "model default"
        if request.num_inference_steps is not None:
            gen_params.num_inference_steps = request.num_inference_steps
        if request.guidance_scale is not None:
            gen_params.guidance_scale = request.guidance_scale
        if request.true_cfg_scale is not None:
            gen_params.true_cfg_scale = request.true_cfg_scale
        if request.seed is None:
            request.seed = random.randint(0, 2**32 - 1)
        gen_params.seed = request.seed
        if request.generator_device is not None:
            gen_params.generator_device = request.generator_device

        logger.info(f"Generating {request.n} image(s) {size_str}")

        request_id = f"img_gen_{self._base_request_id(raw_request)}"
        result = await self._generate_with_async_omni(
            gen_params=gen_params,
            prompt=prompt,
            request_id=request_id,
        )
        images = self._extract_images_from_result(result)
        logger.info(f"Successfully generated {len(images)} image(s)")

        image_data = [ImageData(b64_json=encode_image_base64(img), revised_prompt=None) for img in images]

        return ImageGenerationResponse(
            created=int(time.time()),
            data=image_data,
        )

    async def edit_images(
        self,
        request: ImageEditRequest,
        raw_request: Request | None = None,
    ) -> ImageEditResponse | ErrorResponse:
        """Process image editing request."""
        # 1. Validate output format & compression options
        output_format = self._choose_output_format(request.output_format, request.background)
        if request.response_format != "b64_json":
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail="Only response_format 'b64_json' is supported now.",
            )

        model_name = self._resolve_model_name(raw_request)

        if request.model is not None and model_name is not None and request.model != model_name:
            logger.warning(
                "Model mismatch: request specifies '%s' but server is running '%s'. Using server model.",
                request.model,
                model_name,
            )
        # 2. Build prompt & image inputs
        prompt = OmniTextPrompt(
            prompt=request.prompt,
        )
        if request.negative_prompt is not None:
            prompt["negative_prompt"] = request.negative_prompt
        input_images_list = []
        images = request.image
        urls = request.url
        if images:
            input_images_list.extend(images)
        if urls:
            input_images_list.extend(urls)
        if not input_images_list:
            raise HTTPException(status_code=422, detail="Field 'image' or 'url' is required")
        pil_images = await self._load_input_images(input_images_list)
        prompt["multi_modal_data"] = {}
        prompt["multi_modal_data"]["image"] = pil_images

        # 3. Build sampling params
        gen_params = OmniDiffusionSamplingParams()
        # 3.0 Init with system default values
        app_state_args = None
        if raw_request is not None:
            app_state_args = getattr(raw_request.app.state, "args", None)
        default_sample_param = getattr(app_state_args, "default_sampling_params", None)
        # Currently only have one diffusion stage
        fallback_stage_configs = (
            cast(list[Any] | None, getattr(raw_request.app.state, "stage_configs", None))
            if raw_request is not None
            else None
        )
        stage_types = self._stage_types or self._resolve_stage_types(fallback_stage_configs)
        diffusion_stage_ids = [i for i, t in enumerate(stage_types) if t == "diffusion"]
        if not diffusion_stage_ids:
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail="No diffusion stage configured for image generation.",
            )
        diffusion_stage_id = diffusion_stage_ids[0]
        apply_stage_default_sampling_params(
            default_sample_param,
            gen_params,
            str(diffusion_stage_id),
        )
        if request.n is not None:
            gen_params.num_outputs_per_prompt = request.n
        # 3.1 Parse per-request LoRA (compatible with chat's extra_body.lora shape).
        lora_request, lora_scale = self._parse_lora_request(request.lora)
        if lora_request:
            gen_params.lora_request = lora_request
        if lora_scale is not None:
            gen_params.lora_scale = lora_scale
        # 3.2 Parse and add size if provided
        max_generated_image_size = getattr(app_state_args, "max_generated_image_size", None)

        width, height = None, None
        size = request.size or "auto"
        if size.lower() == "auto":
            width, height = pil_images[0].size  # Use first image size
        else:
            width, height = parse_size(size)

        if max_generated_image_size is not None and (width * height > max_generated_image_size):
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=f"Requested image size {width}x{height} exceeds the maximum allowed "
                f"size of {max_generated_image_size} pixels.",
            )

        if width is not None and height is not None:
            gen_params.width = width
            gen_params.height = height
            size_str = f"{width}x{height}"
        else:
            size_str = "model default"

        # 3.3 Add optional parameters ONLY if provided
        if request.num_inference_steps is not None:
            gen_params.num_inference_steps = request.num_inference_steps
        if request.guidance_scale is not None:
            gen_params.guidance_scale = request.guidance_scale
        if request.true_cfg_scale is not None:
            gen_params.true_cfg_scale = request.true_cfg_scale
        # If seed is not provided, generate a random one to ensure
        # a proper generator is initialized in the backend.
        # This fixes issues where using the default global generator
        # might produce blurry images in some environments.
        if request.seed is None:
            request.seed = random.randint(0, 2**32 - 1)
        gen_params.seed = request.seed
        if request.generator_device is not None:
            gen_params.generator_device = request.generator_device

        # 4. Generate images using AsyncOmni (multi-stage mode)
        request_id = f"img_edit_{self._base_request_id(raw_request)}"
        logger.info(f"Generating {request.n} image(s) {size_str}")
        result = await self._generate_with_async_omni(
            gen_params=gen_params,
            prompt=prompt,
            request_id=request_id,
        )

        # 5. Extract images from result
        images = self._extract_images_from_result(result)
        logger.info(f"Successfully generated {len(images)} image(s)")

        # Encode images to base64
        image_data = [
            ImageData(
                b64_json=self._encode_image_base64_with_compression(
                    img, format=output_format, output_compression=request.output_compression
                ),
                revised_prompt=None,
            )
            for img in images
        ]

        return ImageEditResponse(
            created=int(time.time()),
            data=image_data,
            output_format=output_format,
            size=size_str,
        )
