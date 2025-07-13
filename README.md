# LangChain Prompt Engineering Framework

A modular, extensible framework for advanced prompt engineering, evaluation, and optimization—built entirely on official LangChain APIs.

---

## Why Use This Framework?

Prompt engineering for LLMs is a rapidly evolving field. Ad-hoc scripts and notebooks quickly become unmanageable as you scale up:

- **Templates get duplicated and lost**
- **Prompt evaluation is inconsistent**
- **Switching LLM providers is painful**
- **No easy way to optimize or test prompts at scale**

**This framework solves these problems by providing:**

- **Reusable, composable utilities** for prompt management, LLM selection, evaluation, and optimization
- **A robust CLI** for experimentation, batch testing, and automation
- **A clear separation between reusable logic and demos/scripts**
- **Easy integration** into pipelines, UIs, or production systems

---

## Who Is This For?

- **Prompt engineers** and LLM application developers who want to move beyond notebooks and ad-hoc scripts
- **Teams** standardizing prompt workflows across projects
- **Researchers** running systematic prompt evaluations and ablations
- **Anyone** building LLM-powered products who wants maintainable, testable, and extensible prompt engineering infrastructure

---

## Key Features & Advantages

- **Official LangChain APIs only**: No vendor lock-in, maximum compatibility
- **Multi-provider LLM support**: OpenAI, Anthropic, HuggingFace, and more
- **Prompt template registry**: Centralized, versionable prompt management
- **Built-in evaluation**: Use LangChain's evaluators for single or batch scoring
- **Parameter sweep optimization**: Systematic prompt/LLM parameter tuning
- **CLI and (optional) UI**: Use interactively, in scripts, or as a backend
- **Extensible**: Add your own templates, evaluators, or UI

**Compared to ad-hoc approaches:**  
- No more copy-pasting prompt code  
- Consistent, testable workflows  
- Easy to scale from experimentation to production

---

## Project Structure

```
langchain-prompt-utils/
│
├── langchain_utils/              # Reusable package
│   ├── __init__.py
│   ├── prompt_management.py      # Template registry & management
│   ├── llm_provider.py           # Multi-provider LLM selection
│   ├── evaluator.py              # Evaluation utilities
│   ├── optimizer.py              # Parameter sweep optimization
│   ├── cli.py                    # CLI entry point
│   └── ui.py                     # (optional) Web UI
│
├── examples/                     # Usage demos and workflow scripts
│   └── workflow_integration.py
│
├── tests/                        # Unit/integration tests
│
├── docs/                         # Documentation
│   ├── genai-workflow-integration.md
│   ├── cli_usage.md
│   └── ...
│
├── templates/                    # Prompt templates (JSON)
│
├── .env.example
├── .gitignore
├── LICENSE
├── pyproject.toml
└── README.md
```

---

## Installation

```bash
pip install -e .
```

Set your API keys in `.env` or as environment variables:

```bash
export OPENAI_API_KEY=your-openai-key
export ANTHROPIC_API_KEY=your-anthropic-key
```

---

## CLI Usage

The CLI provides a robust interface for prompt engineering workflows.

### List Templates

```bash
python -m langchain_utils.cli list-templates
```

### Generate Output

From a registered template:

```bash
python -m langchain_utils.cli generate --template summarization --vars '{"text": "LangChain is a framework for LLM apps."}'
```

From a raw template string:

```bash
python -m langchain_utils.cli generate --template "Translate to French: {text}" --vars '{"text": "Hello"}' --raw
```

### Evaluate Output

```bash
python -m langchain_utils.cli evaluate --template summarization --vars '{"text": "LangChain is a framework for LLM apps."}'
```

### Batch Evaluation

Prepare a test cases file (see `docs/cli_usage.md` for format):

```bash
python -m langchain_utils.cli batch-evaluate-cmd --template summarization --test-cases tests/summarization_tests.json
```

### Parameter Sweep Optimization

```bash
python -m langchain_utils.cli optimize --template summarization --test-cases tests/summarization_tests.json --param-grid '{"temperature": [0.3, 0.7], "max_tokens": [128, 256]}'
```

See [docs/cli_usage.md](docs/cli_usage.md) for full details and troubleshooting.

---

## Programmatic Usage

Use the framework as a library in your own pipelines:

```python
from langchain_utils import PromptRegistry, get_llm, evaluate_response

# Load or register templates
registry = PromptRegistry("templates")
prompt = registry.get_template("summarization")

# Get an LLM (OpenAI, Anthropic, HuggingFace, etc.)
llm = get_llm("gpt-3.5-turbo")

# Run a chain
from langchain.chains.llm import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(text="Some input")

# Evaluate the result
score = evaluate_response(llm, "Some input", result)
```

---

## Template Management

- Templates are stored as JSON in the `templates/` directory.
- Register new templates programmatically or by adding files.
- Use the CLI or API to list, retrieve, or use templates.

Example template file:

```json
{
  "template": "Summarize the following text: {text}",
  "input_variables": ["text"]
}
```

---

## Extending the Framework

- **Add new LLM providers**: Extend `llm_provider.py`
- **Custom evaluators**: Add to `evaluator.py`
- **Web UI**: Use `langchain_utils/ui.py` (Streamlit/FastAPI example included)
- **Integrate with LangServe, LangSmith, etc.**: See `docs/genai-workflow-integration.md`

---

## Example: Web UI (Streamlit)

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

## Troubleshooting

- **Invalid JSON for --vars**: Ensure your input variables are a valid JSON string.
- **Template not found**: Make sure your template is registered or use `--raw`.
- **API key errors**: Set your API keys in `.env` or as environment variables.
- **Test cases file errors**: Ensure your test cases file is valid JSON and contains a list of test cases.

---

## Further Reading

- [docs/cli_usage.md](docs/cli_usage.md): Full CLI documentation and examples
- [docs/genai-workflow-integration.md](docs/genai-workflow-integration.md): How this framework fits into GenAI/LLM workflows
- [examples/workflow_integration.py](examples/workflow_integration.py): End-to-end usage demos

---

## License

MIT

---

This framework provides a robust, maintainable, and extensible foundation for prompt engineering with LangChain—whether you're experimenting, testing, or deploying LLM-powered applications.