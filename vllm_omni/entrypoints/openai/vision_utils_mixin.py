"""Shared utilities for vision-based serving classes (image and video generation).

Provides common functionality for:
- Model name resolution
- Request ID generation
- Engine and stage configuration access
"""

from __future__ import annotations

from http import HTTPStatus
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException, Request
from vllm.utils import random_uuid

from vllm_omni.lora.request import LoRARequest
from vllm_omni.lora.utils import stable_lora_int_id

if TYPE_CHECKING:
    from vllm.engine.protocol import EngineClient


class VisionMixin:
    @property
    def engine_client(self) -> EngineClient:
        if hasattr(self, "_engine_client"):
            return self._engine_client  # type: ignore[unknown-attribute]
        raise AttributeError("Engine client not found on this instance.")

    @property
    def model_name(self) -> str:
        if hasattr(self, "_model_name"):
            return self._model_name  # type: ignore[unknown-attribute]
        raise AttributeError("Model name not found on this instance.")

    @staticmethod
    def _base_request_id(raw_request: Request | None, default: str | None = None) -> str:
        """Pull the request id from header if provided, otherwise use default/uuid."""
        if raw_request is not None and ((req_id := raw_request.headers.get("X-Request-Id")) is not None):
            return req_id
        return random_uuid() if default is None else default

    @staticmethod
    def _get_stage_type(stage: Any, default: str = "llm") -> str:
        if isinstance(stage, dict):
            return stage.get("stage_type", default)
        if hasattr(stage, "get"):
            return stage.get("stage_type", default)
        if hasattr(stage, "stage_type"):
            return getattr(stage, "stage_type")
        try:
            return stage["stage_type"] if "stage_type" in stage else default
        except (TypeError, KeyError):
            return default

    @staticmethod
    def _parse_lora_request(lora_body: dict[str, Any] | None) -> tuple[LoRARequest | None, float | None]:
        if lora_body is None:
            return None, None

        if not isinstance(lora_body, dict):
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail="Invalid lora field: expected an object.",
            )

        lora_name = lora_body.get("name") or lora_body.get("lora_name") or lora_body.get("adapter")
        lora_path = (
            lora_body.get("local_path")
            or lora_body.get("path")
            or lora_body.get("lora_path")
            or lora_body.get("lora_local_path")
        )
        lora_scale = lora_body.get("scale")
        if lora_scale is None:
            lora_scale = lora_body.get("lora_scale")

        lora_int_id = lora_body.get("int_id")
        if lora_int_id is None:
            lora_int_id = lora_body.get("lora_int_id")
        if lora_int_id is None and lora_path:
            lora_int_id = stable_lora_int_id(str(lora_path))

        if not lora_name or not lora_path:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail="Invalid lora object: both name and path are required.",
            )

        parsed_lora_scale = float(lora_scale) if lora_scale is not None else None
        return LoRARequest(
            str(lora_name),
            int(lora_int_id),  # type: ignore[arg-type]
            str(lora_path),
        ), parsed_lora_scale

    @staticmethod
    def _extract_images_from_result(result: Any) -> list[Any]:
        images: list[Any] = []
        if hasattr(result, "images") and result.images:
            images = result.images
        elif hasattr(result, "request_output"):
            request_output = result.request_output
            if isinstance(request_output, dict):
                if request_output.get("images"):
                    images = request_output["images"]
            elif hasattr(request_output, "images") and request_output.images:
                images = request_output.images
        return images

    def _resolve_model_name(self, raw_request: Request | None) -> str | None:
        if self.model_name:
            return self.model_name
        if raw_request is None:
            return None
        serving_models = getattr(raw_request.app.state, "openai_serving_models", None)
        if serving_models and getattr(serving_models, "base_model_paths", None):
            base_paths = serving_models.base_model_paths
            if base_paths:
                return base_paths[0].name
        return None
