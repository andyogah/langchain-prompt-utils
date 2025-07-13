# LangChain Utils CLI Usage Guide

This guide explains how to use the `langchain_utils` command-line interface (CLI) for prompt engineering, including prompt generation, evaluation, batch testing, optimization, and template management.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Templates](#templates)
- [CLI Commands](#cli-commands)
  - [List Templates](#list-templates)
  - [Generate Output](#generate-output)
  - [Evaluate Output](#evaluate-output)
  - [Batch Evaluation](#batch-evaluation)
  - [Parameter Sweep Optimization](#parameter-sweep-optimization)
- [Raw vs Registered Templates](#raw-vs-registered-templates)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

---

## Overview

The `langchain_utils` CLI provides a flexible interface for:

- Generating LLM outputs from prompt templates
- Evaluating outputs using built-in evaluators
- Batch testing and optimization
- Managing prompt templates

You can use either registered templates (from your `templates/` directory) or raw template strings.

---

## Installation

Make sure you have installed your project in editable mode:

```bash
pip install -e .
```

Set your API keys in `.env` or as environment variables:

```bash
export OPENAI_API_KEY=your-openai-key
export ANTHROPIC_API_KEY=your-anthropic-key
```

---

## Templates

Templates are stored as JSON files in the `templates/` directory. Example:

```json
{
  "template": "Summarize the following text: {text}",
  "input_variables": ["text"]
}
```

---

## CLI Commands

### List Templates

List all available registered templates:

```bash
python -m langchain_utils.cli list-templates
```

---

### Generate Output

Generate output from a prompt template and input variables:

```bash
python -m langchain_utils.cli generate --template summarization --vars '{"text": "LangChain is a framework for developing LLM-powered applications."}'
```

Use a raw template string (not registered):

```bash
python -m langchain_utils.cli generate --template "Translate to French: {text}" --vars '{"text": "Hello"}' --raw
```

---

### Evaluate Output

Evaluate a prompt template and input variables using built-in evaluators:

```bash
python -m langchain_utils.cli evaluate --template summarization --vars '{"text": "LangChain is a framework for developing LLM-powered applications."}'
```

---

### Batch Evaluation

Evaluate a prompt template on multiple test cases:

1. Create a test cases file (e.g., `tests/summarization_tests.json`):

    ```json
    [
      {
        "input": {"text": "LangChain is a framework for developing LLM-powered applications."},
        "prediction": "LangChain helps build apps with large language models.",
        "reference": "LangChain is used to build LLM apps."
      },
      {
        "input": {"text": "OpenAI's GPT-3 is a powerful language model."},
        "prediction": "GPT-3 is a strong language model from OpenAI.",
        "reference": "GPT-3 is a language model by OpenAI."
      }
    ]
    ```

2. Run batch evaluation:

    ```bash
    python -m langchain_utils.cli batch-evaluate-cmd --template summarization --test-cases tests/summarization_tests.json
    ```

---

### Parameter Sweep Optimization

Optimize prompt parameters (e.g., temperature, max_tokens):

```bash
python -m langchain_utils.cli optimize --template summarization --test-cases tests/summarization_tests.json --param-grid '{"temperature": [0.3, 0.7], "max_tokens": [128, 256]}'
```

---

## Raw vs Registered Templates

- **Registered templates**: Stored in `templates/` and referenced by name.
- **Raw templates**: Passed directly as a string with `--raw`.

---

## Examples

**Generate with registered template:**

```bash
python -m langchain_utils.cli generate --template summarization --vars '{"text": "Your text here"}'
```

**Generate with raw template:**

```bash
python -m langchain_utils.cli generate --template "Say hello to {name}" --vars '{"name": "Andre"}' --raw
```

**Evaluate:**

```bash
python -m langchain_utils.cli evaluate --template summarization --vars '{"text": "Your text here"}'
```

**Batch evaluate:**

```bash
python -m langchain_utils.cli batch-evaluate-cmd --template summarization --test-cases tests/summarization_tests.json
```

**Optimize:**

```bash
python -m langchain_utils.cli optimize --template summarization --test-cases tests/summarization_tests.json --param-grid '{"temperature": [0.3, 0.7], "max_tokens": [128, 256]}'
```

---

## Troubleshooting

- **Invalid JSON for --vars**: Ensure your input variables are a valid JSON string.
- **Template not found**: Make sure your template is registered in the `templates/` directory or use `--raw`.
- **API key errors**: Set your API keys in `.env` or as environment variables.
- **Test cases file errors**: Ensure your test cases file is valid JSON and contains a list of test cases.

---

## Help

For help on any command, run:

```bash
python -m langchain_utils.cli --help
python -m langchain_utils.cli generate --help
```

---

This CLI enables robust, flexible prompt engineering workflows for both experimentation and production.
