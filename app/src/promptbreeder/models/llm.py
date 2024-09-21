import yaml
from oocl_opencompass.models import AzureOpenAI, QwenAPI, TgiAPI


class LLM:
    def __init__(self, config_path):
        self.config_path = config_path
        with open(self.config_path, "r") as file:
            self.cfg = yaml.safe_load(file)

        self.meta_template = dict(
            round=[
                dict(role="HUMAN", api_role="HUMAN"),
                dict(role="BOT", api_role="BOT", generate=True),
            ],
            reserved_roles=[
                dict(role="SYSTEM", api_role="SYSTEM"),
            ],
        )
        self.default_configs = self.cfg["models"]

    def initialize_llm(self, model_name, customized_config):
        kwargs = {
            **self.default_configs[model_name],
            **{"meta_template": self.meta_template},
            **customized_config,
        }
        path = kwargs["path"]

        print(f"Initializing model {model_name} with config: {kwargs}")

        if path.startswith("qwen"):
            return QwenAPI(**kwargs)
        elif path.startswith("gpt"):
            return AzureOpenAI(**kwargs)
        elif path.startswith("Llama"):
            return TgiAPI(**kwargs)

    def get_llm(self, llm_name, customized_config=None):
        customized_config = customized_config or {}
        llm_model = self.initialize_llm(llm_name, customized_config)
        return llm_model


if __name__ == "__main__":
    config_path = "/workspace/promptbreeder/config/llm.yml"
    customized_config = {"temperature": 1}
    selected_llm = LLM(config_path).get_llm("gpt-4", customized_config)

"""
python /workspace/promptbreeder/promptbreeder/models/llm.py
"""
