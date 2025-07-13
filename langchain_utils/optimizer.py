"""
Prompt optimization utilities for LangChain workflows.

Provides:
- Parameter sweep for LLM temperature, max_tokens, etc.
"""

from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

def parameter_sweep(llm, prompt_template: PromptTemplate, test_cases, param_grid):
    """
    Perform a parameter sweep over LLM parameters (e.g., temperature, max_tokens).

    Args:
        llm: LangChain LLM instance
        prompt_template: PromptTemplate instance
        test_cases: List of test case dicts with 'input' and 'evaluator'
        param_grid: Dict with parameter lists, e.g., {"temperature": [0.3, 0.7], "max_tokens": [128, 256]}

    Returns:
        Dict with best parameters, best score, and all results
    """
    best_score = float('-inf')
    best_params = None
    results = []
    for temperature in param_grid.get("temperature", [0.7]):
        for max_tokens in param_grid.get("max_tokens", [256]):
            llm.temperature = temperature
            llm.max_tokens = max_tokens
            chain = LLMChain(llm=llm, prompt=prompt_template)
            scores = []
            for case in test_cases:
                output = chain.run(**case["input"])
                # Assume you have an evaluator function
                score = case.get("evaluator")(case["input"], output, case.get("reference"))
                scores.append(score)
            avg_score = sum(scores) / len(scores)
            results.append({"params": {"temperature": temperature, "max_tokens": max_tokens}, "score": avg_score})
            if avg_score > best_score:
                best_score = avg_score
                best_params = {"temperature": temperature, "max_tokens": max_tokens}
    return {"best_params": best_params, "best_score": best_score, "results": results}
