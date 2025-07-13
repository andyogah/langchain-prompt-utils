"""
Command-line interface for prompt engineering workflows using langchain_utils.

Provides commands for:
- Prompt generation (single or batch)
- Evaluation (single or batch)
- Template listing
- Parameter sweep optimization
"""

import json
import click
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_utils.prompt_management import PromptRegistry
from langchain_utils.llm_provider import get_llm
from langchain_utils.evaluator import evaluate_response, batch_evaluate
from langchain_utils.optimizer import parameter_sweep


@click.group()
def cli():
    """LangChain Prompt Engineering CLI."""
    pass

@cli.command()
@click.option("--template", required=True, help="Template name or raw template string")
@click.option("--vars", required=True, help="JSON string of input variables")
@click.option("--model", default="gpt-3.5-turbo", show_default=True)
@click.option("--raw/--registered", default=False, help="Use raw template string or registered template")
def generate(template, vars, model, raw):
    """
    Generate output from a prompt template and input variables.
    """
    try:
        variables = json.loads(vars)
    except Exception as e:
        click.echo(f"Invalid JSON for --vars: {e}")
        return
    if raw:
        prompt = PromptTemplate(template=template, input_variables=list(variables.keys()))
    else:
        registry = PromptRegistry()
        prompt = registry.get_template(template)
        if not prompt:
            click.echo(f"Template '{template}' not found in registry.")
            return
    llm = get_llm(model)
    chain = LLMChain(llm=llm, prompt=prompt)
    try:
        result = chain.run(**variables)
        click.echo(result)
    except Exception as e:
        click.echo(f"Error: {e}")

@cli.command()
@click.option("--template", required=True, help="Template name or raw template string")
@click.option("--vars", required=True, help="JSON string of input variables")
@click.option("--model", default="gpt-3.5-turbo", show_default=True)
@click.option("--raw/--registered", default=False, help="Use raw template string or registered template")
def evaluate(template, vars, model, raw):
    """
    Evaluate a prompt template and input variables using built-in evaluators.
    """
    try:
        variables = json.loads(vars)
    except Exception as e:
        click.echo(f"Invalid JSON for --vars: {e}")
        return
    if raw:
        prompt = PromptTemplate(template=template, input_variables=list(variables.keys()))
    else:
        registry = PromptRegistry()
        prompt = registry.get_template(template)
        if not prompt:
            click.echo(f"Template '{template}' not found in registry.")
            return
    llm = get_llm(model)
    chain = LLMChain(llm=llm, prompt=prompt)
    try:
        prediction = chain.run(**variables)
        result = evaluate_response(llm, " ".join(str(v) for v in variables.values()), prediction)
        click.echo(json.dumps(result, indent=2))
    except Exception as e:
        click.echo(f"Error: {e}")

@cli.command()
def list_templates():
    """
    List all available registered templates.
    """
    registry = PromptRegistry()
    templates = list(registry.templates.keys()) + list(registry.chat_templates.keys())
    if not templates:
        click.echo("No templates found.")
    else:
        click.echo("Available templates:")
        for name in templates:
            click.echo(f"- {name}")

@cli.command()
@click.option("--template", required=True, help="Template name or raw template string")
@click.option("--test-cases", required=True, help="Path to JSON file with a list of test cases")
@click.option("--model", default="gpt-3.5-turbo", show_default=True)
@click.option("--raw/--registered", default=False, help="Use raw template string or registered template")
def batch_evaluate_cmd(template, test_cases, model, raw):
    """
    Batch evaluate a prompt template on multiple test cases.
    Each test case should be a dict with 'input', 'prediction', and optionally 'reference'.
    """
    try:
        with open(test_cases, "r") as f:
            cases = json.load(f)
    except Exception as e:
        click.echo(f"Failed to load test cases: {e}")
        return
    if not isinstance(cases, list):
        click.echo("Test cases file must contain a list of test cases.")
        return
    if raw:
        if not cases:
            click.echo("No test cases provided.")
            return
        input_vars = list(cases[0].get("input", {}).keys())
        prompt = PromptTemplate(template=template, input_variables=input_vars)
    else:
        registry = PromptRegistry()
        prompt = registry.get_template(template)
        if not prompt:
            click.echo(f"Template '{template}' not found in registry.")
            return
    llm = get_llm(model)
    chain = LLMChain(llm=llm, prompt=prompt)
    results = []
    for case in cases:
        try:
            prediction = chain.run(**case["input"])
            result = evaluate_response(llm, " ".join(str(v) for v in case["input"].values()), prediction, case.get("reference"))
            results.append({"input": case["input"], "prediction": prediction, "evaluation": result})
        except Exception as e:
            results.append({"input": case.get("input"), "error": str(e)})
    click.echo(json.dumps(results, indent=2))

@cli.command()
@click.option("--template", required=True, help="Template name or raw template string")
@click.option("--test-cases", required=True, help="Path to JSON file with a list of test cases")
@click.option("--model", default="gpt-3.5-turbo", show_default=True)
@click.option("--raw/--registered", default=False, help="Use raw template string or registered template")
@click.option("--param-grid", required=True, help="JSON string of parameter grid, e.g. '{\"temperature\": [0.3, 0.7], \"max_tokens\": [128, 256]}'")
def optimize(template, test_cases, model, raw, param_grid):
    """
    Perform parameter sweep optimization for a prompt template on test cases.
    """
    try:
        with open(test_cases, "r") as f:
            cases = json.load(f)
        param_grid_dict = json.loads(param_grid)
    except Exception as e:
        click.echo(f"Failed to load test cases or param grid: {e}")
        return
    if not isinstance(cases, list):
        click.echo("Test cases file must contain a list of test cases.")
        return
    if raw:
        if not cases:
            click.echo("No test cases provided.")
            return
        input_vars = list(cases[0].get("input", {}).keys())
        prompt = PromptTemplate(template=template, input_variables=input_vars)
    else:
        registry = PromptRegistry()
        prompt = registry.get_template(template)
        if not prompt:
            click.echo(f"Template '{template}' not found in registry.")
            return
    llm = get_llm(model)
    # Each test case must have an 'evaluator' function; for CLI, use built-in evaluator
    def default_evaluator(input_vars, prediction, reference=None):
        return evaluate_response(llm, " ".join(str(v) for v in input_vars.values()), prediction, reference).get("score", 0)
    for case in cases:
        case["evaluator"] = default_evaluator
    result = parameter_sweep(llm, prompt, cases, param_grid_dict)
    click.echo(json.dumps(result, indent=2))

if __name__ == "__main__":
    cli()