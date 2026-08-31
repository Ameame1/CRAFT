"""
CRAFT Client - HTTP client for CRAFT model server.

Provides an interface to call the CRAFT model deployed via vLLM.
"""

from typing import Dict, List, Optional

from openai import OpenAI

from src.templates import get_template
from src.craft.extracts import extract_answer


class CRAFTClient:
    """Client for the CRAFT model server."""

    def __init__(
        self,
        base_url: str = "http://localhost:8002/v1",
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
        template_version: str = "v1",
    ):
        """
        Initialize the CRAFT client.

        Args:
            base_url: vLLM server URL (e.g., "http://localhost:8002/v1")
            model_name: Model name (auto-detected if not provided)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            template_version: Prompt template version (v1-v5)
        """
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.template_version = template_version

        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=base_url,
            api_key="not-needed"  # vLLM doesn't require API key
        )

        # Auto-detect model name if not provided
        if model_name:
            self.model_name = model_name
        else:
            models = self.client.models.list()
            if models.data:
                self.model_name = models.data[0].id
            else:
                raise ValueError(f"No models found at {base_url}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max_tokens

        Returns:
            Generated response text
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            top_p=self.top_p,
        )
        return response.choices[0].message.content

    def generate(self, question: str, documents: List[Dict]) -> Dict:
        """
        Generate a response for the given question and documents.

        Args:
            question: The question to answer
            documents: List of document dicts with 'title' and 'contents'/'text'

        Returns:
            Dict with:
                - response: Raw model response
                - answer: Extracted answer
        """
        # Format documents
        docs_text = self._format_documents(documents)

        # Get the template and format the prompt
        template = get_template(self.template_version)
        prompt = template.format(query=question, docs=docs_text)

        messages = [
            {"role": "user", "content": prompt},
        ]

        response = self.chat(messages)

        return {
            "response": response,
            "answer": extract_answer(response),
        }

    def _format_documents(self, documents: List[Dict]) -> str:
        """Format documents for the prompt."""
        doc_strings = []
        for i, doc in enumerate(documents, 1):
            title = doc.get('title', '')
            content = doc.get('contents', doc.get('text', ''))
            doc_strings.append(f"{i}. {title}: {content}")
        return "\n".join(doc_strings)


def check_craft_server_ready(base_url: str = "http://localhost:8002/v1") -> bool:
    """
    Check if the CRAFT server is ready.

    Args:
        base_url: Server URL (e.g., "http://localhost:8002/v1")

    Returns:
        True if server is responding, False otherwise
    """
    try:
        client = OpenAI(base_url=base_url, api_key="not-needed")
        models = client.models.list()
        return len(models.data) > 0
    except Exception:
        return False


def load_craft_client(
    base_url: str = "http://localhost:8002/v1",
    **kwargs
) -> CRAFTClient:
    """
    Load CRAFT client, checking server availability first.

    Args:
        base_url: vLLM server URL
        **kwargs: Additional arguments for CRAFTClient

    Returns:
        CRAFTClient instance

    Raises:
        RuntimeError: If server is not available
    """
    if not check_craft_server_ready(base_url):
        raise RuntimeError(f"CRAFT server not available at {base_url}")
    return CRAFTClient(base_url=base_url, **kwargs)
