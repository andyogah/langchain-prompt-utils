"""
Evaluation utilities for LangChain prompt engineering.

Provides:
- Single response evaluation using built-in LangChain evaluators
- Batch evaluation for multiple test cases
"""

from langchain.evaluation import load_evaluator, EvaluatorType

def evaluate_response(llm, input_text, prediction, reference=None, criteria="helpfulness"):
    """
    Evaluate a single LLM response using LangChain's built-in evaluators.

    Args:
        llm: LangChain LLM instance
        input_text: The input prompt
        prediction: The LLM's output
        reference: (Optional) Reference/ground truth output
        criteria: Evaluation criteria (default: "helpfulness")

    Returns:
        Evaluation result dict
    """
    evaluator = load_evaluator(EvaluatorType.CRITERIA, llm=llm, criteria=criteria)
    return evaluator.evaluate_strings(
        input=input_text,
        prediction=prediction,
        reference=reference
    )

def batch_evaluate(llm, test_cases, criteria="helpfulness"):
    """
    Evaluate a batch of test cases.

    Args:
        llm: LangChain LLM instance
        test_cases: List of dicts with keys 'input', 'prediction', and optionally 'reference'
        criteria: Evaluation criteria

    Returns:
        List of evaluation result dicts
    """
    results = []
    for case in test_cases:
        result = evaluate_response(
            llm,
            case["input"],
            case["prediction"],
            case.get("reference"),
            criteria=criteria
        )
        results.append(result)
    return results
