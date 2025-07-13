"""
Multi-provider LLM selection utility for LangChain.

Supports OpenAI, Anthropic, and HuggingFace models.
"""

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceHub

def get_llm(model_name: str):
    """
    Return a LangChain LLM/chat model instance for the given model name.

    Supports:
    - OpenAI (model names starting with 'gpt-')
    - Anthropic (model names starting with 'claude')
    - HuggingFace (model names starting with 'hf-' or 'huggingface-')
    """
    if model_name.startswith("gpt-"):
        return ChatOpenAI(model=model_name)
    elif model_name.startswith("claude"):
        return ChatAnthropic(model=model_name)
    elif model_name.startswith("hf-") or model_name.startswith("huggingface"):
        return HuggingFaceHub(repo_id=model_name.replace("hf-", "").replace("huggingface-", ""))
    raise ValueError(f"Unknown model: {model_name}")
