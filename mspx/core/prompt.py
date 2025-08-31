#

# input/prompt formatter

__all__ = [
    "InputFormatter",
]

from collections import OrderedDict

class TemplateAlpaca:
    T0 = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
    T1 = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"

    def __repr__(self):
        return f"Alpaca:\n{self.T0}\n{self.T1}"

    def format(self, **kwargs):
        t = self.T1 if kwargs.get("input") else self.T0
        return t.format(**kwargs)

class TemplateTulu:
    def __repr__(self):
        return f"Tulu:<|system|><|user|><|assistant|>"

    def format(self, messages=None, system="", prompt="", _return_list=False):
        if messages is None:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
        # --
        all_pieces = []
        for mid, mm in enumerate(messages):
            if mm["role"] == "system":
                all_pieces.append("<|system|>\n" + mm["content"].strip() + "\n")
            elif mm["role"] == "user":  # append assistant prefix
                all_pieces.append("<|user|>\n" + mm["content"].strip() + "\n<|assistant|>\n")
                assert mid+1 == len(messages) or messages[mid+1]["role"] == "assistant"
            elif mm["role"] == "assistant":  # previous one must be user's query
                assert mid>0 and messages[mid-1]["role"] == "user"
                all_pieces.append(mm["content"].strip() + "\n")
            else:
                raise ValueError("Invalid role: {}".format(mm["role"]))
        if _return_list:
            return all_pieces
        else:
            return "".join(all_pieces)

INFO_PROMPTS = {
    "plain": {"template": "{prompt}", "stop": [], "required_input_keys": ["prompt"]},  # default one
    "alpaca": {"template": TemplateAlpaca(), "stop": ["###"], "required_input_keys": ["instruction", "input"]},
    "tulu2": {"template": TemplateTulu(), "stop": ["\n<user>"], "required_input_keys": ["system", "prompt"]},
}

class InputFormatter:
    def __init__(self, input_format: str, stop_seqs=()):
        self.input_format = input_format
        self.name = self.input_format.split(":")[0]
        # --
        _name = self.name
        _info = INFO_PROMPTS[_name]
        _stop_tokens = _info["stop"] + list(stop_seqs)
        self._stop_tokens = list(OrderedDict([(z, 1) for z in _stop_tokens]).keys())
        self._template = _info["template"]
        self._required_input_keys = _info["required_input_keys"]

    @property
    def stop_tokens(self):
        return self._stop_tokens

    @property
    def template(self):
        return self._template

    @property
    def required_input_keys(self):
        return self._required_input_keys

    def format(self, _m=None, **kwargs) -> str:
        if _m is None:
            _m = {}
        _m.update(kwargs)
        ret = self._template.format(**_m)
        return ret
