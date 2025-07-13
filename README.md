# LangChain-Based Prompt Engineering Framework

This repo demonstrates advanced prompt engineering using only official LangChain APIs, with reusable logic in the `langchain_utils` package.

---

## Project File Structure

```
/c:/Users/andre/ai-models/prompt-engineering/
│
├── langchain_utils/              # Your reusable package
│   ├── __init__.py
│   ├── prompt_management.py
│   ├── llm_provider.py
│   ├── evaluator.py
│   ├── optimizer.py
│   ├── cli.py
│   └── ui.py                     # (optional, for web UI)
│
├── examples/                     # Usage demos and workflow scripts
│   ├── workflow_integration.py
│   └── ... (other demos)
│
├── tests/                        # Unit and integration tests
│   ├── test_workflow_integration.py
│   └── ... (other tests)
│
├── docs/                         # Documentation
│   ├── genai-workflow-integration.md
│   └── ... (other docs)
│
├── templates/                    # Prompt templates (if used)
│   └── ... (template files)
│
├── .env.example
├── .gitignore
├── LICENSE
├── pyproject.toml
└── README.md
```

---

## Usage Overview

To maximize flexibility and reusability:

1. **All reusable logic (prompt management, LLM provider, evaluation, optimization, etc.) is in the `langchain_utils/` package.**
2. **You can use it as a library in pipelines, as a CLI, or as a backend for a UI.**
3. **Only runnable demos and usage examples are in `examples/`.**

---

## 1. Programmatic Usage in a Pipeline

```python
from langchain_utils import PromptRegistry, get_llm, evaluate_response

# Load or register templates
registry = PromptRegistry("templates")
prompt = registry.get_template("summarization")

# Get an LLM
llm = get_llm("gpt-3.5-turbo")

# Run a chain
from langchain.chains.llm import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(text="Some input")

# Evaluate the result
score = evaluate_response(llm, "Some input", result)

# Pass result to downstream process
```

---

## 2. CLI Usage

A CLI is provided via `langchain_utils/cli.py`:

```bash
python -m langchain_utils.cli generate --template "Summarize the following: {text}" --text "Your text here"
```

You can extend the CLI with more commands for evaluation, optimization, etc.

---

## 3. Minimal Web UI or API (Optional)

You can add a FastAPI or Streamlit app in `langchain_utils/ui.py` for interactive use.  
Example (Streamlit):

```python
# langchain_utils/ui.py
import streamlit as st
from langchain_utils import PromptRegistry, get_llm

st.title("Prompt Engineering Playground")
prompt_text = st.text_area("Prompt Template", "Summarize the following: {text}")
user_input = st.text_area("Input Text", "")
if st.button("Generate"):
    registry = PromptRegistry()
    registry.register_template("custom", {"template": prompt_text, "input_variables": ["text"]})
    prompt = registry.get_template("custom")
    llm = get_llm("gpt-3.5-turbo")
    from langchain.chains.llm import LLMChain
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(text=user_input)
    st.write(result)
```

---

## 4. Keep Only Demos in `examples/`

All reusable code is in `langchain_utils/`.  
Keep only minimal scripts in `examples/` that show how to use the package.

---

## 5. Summary

- All reusable logic is in `langchain_utils/` (importable as a library).
- Use the CLI for standalone/command-line use.
- Optionally add a UI for interactive use.
- Use in pipelines by importing and composing the API.
- Only keep demos in `examples/`.

---

This structure makes your codebase usable as a library, CLI, or UI, and easy to integrate into any pipeline or product.