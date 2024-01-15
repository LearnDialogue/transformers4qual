from ..configuration_utils import PretrainedConfig
from ..modeling_utils import PreTrainedModel
from .base import Pipeline


class LangchainConfig(PretrainedConfig):

    model_type = "langchain"

    def __init__(self, runnable, **kwargs):
        self.runnable = runnable
        super().__init__(**kwargs)


class LangchainModelForProxyLLM(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.runnable = config.runnable

    def forward(self, model_input):
        return self.runnable.invoke(model_input)


class LLMProxyPipeline(Pipeline):
    def _sanitize_parameters(self, **pipeline_parameters):
        return pipeline_parameters, pipeline_parameters, pipeline_parameters

    def preprocess(self, inputs, **kwargs):
        return {"model_input": inputs}

    def _forward(self, model_inputs, **kwargs):
        output = self.model(**model_inputs)
        return {"output": output}

    def postprocess(self, output, *_):
        return {"output": output}
