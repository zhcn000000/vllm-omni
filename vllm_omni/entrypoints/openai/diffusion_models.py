from vllm.entrypoints.openai.engine.protocol import (
    ModelCard,
    ModelList,
    ModelPermission,
)
from vllm.entrypoints.openai.models.protocol import BaseModelPath


class DiffusionServingModels:
    """Minimal OpenAIServingModels implementation for diffusion-only servers.

    vLLM's /v1/models route expects `app.state.openai_serving_models` to expose
    `show_available_models()`. In pure diffusion mode we don't initialize the
    full OpenAIServingModels (it depends on LLM-specific processors), so we
    provide a lightweight fallback.
    """

    def __init__(self, base_model_paths: list[BaseModelPath]) -> None:
        self._base_model_paths = base_model_paths
        self.model_name = base_model_paths[0].name

    async def show_available_models(self) -> ModelList:
        return ModelList(
            data=[
                ModelCard(
                    id=base_model.name,
                    root=base_model.model_path,
                    permission=[ModelPermission()],
                )
                for base_model in self._base_model_paths
            ]
        )
