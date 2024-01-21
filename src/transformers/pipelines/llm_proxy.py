import asyncio

from langchain_core.messages.ai import AIMessage
from langchain.callbacks import get_openai_callback

from ..configuration_utils import PretrainedConfig
from ..modeling_utils import PreTrainedModel
from .base import Pipeline


class LangchainConfig(PretrainedConfig):
    def __init__(self, runnable, mock_llm_call=False, **kwargs):
        self.runnable = runnable
        self.mock_llm_call = mock_llm_call
        super().__init__(**kwargs)


class LangchainModelForProxyLLM(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.runnable = config.runnable
    
    async def _forward_concurrent(self, model_input):
        return await asyncio.gather(
            *[
                self.runnable.ainvoke({k: v[i] for k, v in model_input.items()})
                for i in range(len(model_input[list(model_input.keys())[0]]))
            ]
        )

    def forward(self, model_input):
        if self.config.mock_llm_call:
            return [
                f"The score is {random.randint(-1, 8)}. This was generated from a mock call."
                for i in range(len(model_input[list(model_input.keys())[0]]))
            ]
        return asyncio.run(self._forward_concurrent(model_input))


class LLMProxyPipeline(Pipeline):
    def _sanitize_parameters(self, **pipeline_parameters):
        return pipeline_parameters, pipeline_parameters, pipeline_parameters

    def preprocess(self, inputs, **kwargs):
        return {"model_input": inputs}

    def _forward(self, model_inputs, **kwargs):
        print(f"Executing Langchain runnable.")
        output = self.model(**model_inputs)
        print(f"Finished executing Langchain runnable.")
        return output

    def postprocess(self, output, **kwargs):
        return output
