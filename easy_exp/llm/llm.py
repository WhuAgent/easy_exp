import os
import json
import yaml
from easy_exp.llm.message import Message
from openai import OpenAI

llm_config_path = os.path.join(os.getcwd(), 'easy_exp/llm/config.yaml')
with open(llm_config_path, "r", encoding="UTF-8") as f:
    llm_config = yaml.safe_load(f)
    
sum_prompt_token = 0
sum_completion_token = 0
sum_total_token = 0

    
def remove_unnecessary_keys(**kwargs):
    keys = ["api_key", "base_url", "model"]
    for key in keys:
        if key in kwargs.keys():
            del kwargs[key]
    return kwargs


def chat_llm_json(messages, **kwargs):
    kwargs["response_format"] = {"type": "json_object"}
    return chat_llm(messages, **kwargs)
    

def chat_llm(messages, **kwargs):
    global sum_prompt_token, sum_completion_token, sum_total_token
    api_key = get_api_key(**kwargs)
    base_url = get_base_url(**kwargs)
    model = get_model(**kwargs)
    kwargs = remove_unnecessary_keys(**kwargs)
    
    openai_client = OpenAI(api_key=api_key, base_url=base_url)
    openai_messages = []
    
    for message in messages:
        if isinstance(message, Message):
            openai_messages.append(message.to_openai_message())
        else:
            openai_messages.append(message)

    response = openai_client.chat.completions.create(
        messages=openai_messages,
        model=model,
        **kwargs
    )
    
    response_text = ""
    prompt_tokens = 0
    completion_tokens = 0
    if "stream" in kwargs and kwargs["stream"]:
        for chunk in response:
            if len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                response_text += chunk.choices[0].delta.content
                if chunk.usage:
                    prompt_tokens += chunk.usage.prompt_tokens
                    completion_tokens += chunk.usage.completion_tokens
    else:
        response_text = response.choices[0].message.content
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        sum_prompt_token += prompt_tokens
        sum_completion_token += completion_tokens
        sum_total_token += prompt_tokens + completion_tokens
    
    if kwargs.get("response_format") == {"type": "json_object"}:
        response_text = json.loads(response_text)
        
    openai_messages.append({"role": "assistant", "content": response_text})
    print(json.dumps(openai_messages, indent=4, ensure_ascii=False))

    return response_text


def get_model_family(model):
    if "openai" in model or "gpt" in model:
        return "openai"
    if "deepseek" in model:
        return "deepseek"
    if "qwen" in model or "qwq" in model:
        return "qwen"
    return "openai"


def get_api_key(**kwargs):
    if "api_key" in kwargs.keys():
        api_key = kwargs.get("api_key")
    else:
        model_family = get_model_family(get_model(**kwargs))
        api_key = llm_config.get(model_family).get("api_key", os.getenv("OPENAI_API_KEY"))

    return api_key


def get_base_url(**kwargs):
    if "base_url" in kwargs.keys():
        base_url = kwargs.get("base_url")
    else:
        model_family = get_model_family(get_model(**kwargs))
        base_url = llm_config.get(model_family).get("base_url",
                                                    os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1/"))

    return base_url


def get_model(**kwargs):
    if "model" in kwargs.keys():
        model = kwargs.get("model")
    else:
        model = llm_config.get("default_model", os.getenv("OPENAI_MODEL"))

    return model

def init():
    global sum_prompt_token, sum_completion_token, sum_total_token
    sum_prompt_token = 0
    sum_completion_token = 0
    sum_total_token = 0

def report():
    return sum_prompt_token, sum_completion_token, sum_total_token