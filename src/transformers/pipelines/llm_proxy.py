from langchain_core.messages.ai import AIMessage

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

    def forward(self, model_input):
        if self.config.mock_llm_call:
            return [
                {"label": "3.0"}
                for i in range(len(model_input[list(model_input.keys())[0]]))
            ]
        return [
            self.runnable.invoke({k: v[i] for k, v in model_input.items()})
            for i in range(len(model_input[list(model_input.keys())[0]]))
        ]


class LLMProxyPipeline(Pipeline):
    def _sanitize_parameters(self, **pipeline_parameters):
        return pipeline_parameters, pipeline_parameters, pipeline_parameters

    def preprocess(self, inputs, **kwargs):
        return {"model_input": inputs}

    def _forward(self, model_inputs, **kwargs):
        output = self.model(**model_inputs)
        return output

    def postprocess(self, output, *_):
        return output
