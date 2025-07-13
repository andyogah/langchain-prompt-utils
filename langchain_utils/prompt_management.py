"""
Prompt management and registry for LangChain-based workflows.

Provides:
- Loading and registering prompt templates (standard and chat)
- Formatting prompts for use in chains
"""

from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import json
from pathlib import Path

class PromptRegistry:
    """
    Registry for managing prompt templates (standard and chat) for LangChain workflows.

    Loads templates from a directory, allows registration, retrieval, and formatting.
    """
    def __init__(self, templates_dir: str = "templates"):
        self.templates_dir = Path(templates_dir)
        self.templates = {}
        self.chat_templates = {}
        self.load_templates()

    def load_templates(self):
        """Load prompt templates from the templates directory."""
        if not self.templates_dir.exists():
            return
        for template_file in self.templates_dir.glob("*.json"):
            with open(template_file, 'r') as f:
                template_data = json.load(f)
                self.register_template(template_file.stem, template_data)

    def register_template(self, name: str, template_data: dict):
        """Register a prompt template by name and template data."""
        if template_data.get("type") == "chat":
            messages = []
            for msg in template_data["messages"]:
                if msg["role"] == "system":
                    messages.append(SystemMessagePromptTemplate.from_template(msg["content"]))
                elif msg["role"] == "human":
                    messages.append(HumanMessagePromptTemplate.from_template(msg["content"]))
            self.chat_templates[name] = ChatPromptTemplate.from_messages(messages)
        else:
            self.templates[name] = PromptTemplate(
                template=template_data["template"],
                input_variables=template_data.get("input_variables", [])
            )

    def get_template(self, name: str):
        """Retrieve a prompt template by name."""
        return self.templates.get(name) or self.chat_templates.get(name)

    def format_prompt(self, name: str, **kwargs):
        """
        Format a prompt template with variables.

        Returns a formatted string or list of messages for chat templates.
        """
        template = self.get_template(name)
        if not template:
            raise ValueError(f"Template '{name}' not found")
        if isinstance(template, ChatPromptTemplate):
            return template.format_messages(**kwargs)
        return template.format(**kwargs)
