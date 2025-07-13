"""
Minimal web UI (Streamlit) for interactive prompt engineering using langchain_utils.

Allows users to submit prompts and input text, and view LLM results.
"""

from langchain_utils import PromptRegistry, get_llm, evaluate_response

registry = PromptRegistry("templates")
prompt = registry.get_template("summarization")
llm = get_llm("gpt-3.5-turbo")
from langchain.chains.llm import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(text="Some input")
# Pass result to downstream process