"""Query OpenAI Language Models.

Functions for using OpenAI's API to query language models.

Typical usage example:
    ```
    verify_openai_access(...)
    model_settings = OpenAI_Model_Settings(...)
    res = call_openai_api(
        "Q: How many legs does a cat have",
        model_settings
        )
    ```
"""

import openai
from file_IO_handler import get_plaintext_file_contents
import pathlib

ENGINES = [
    "text-ada-001",
    "text-babbage-001",
    "text-curie-001",
    "text-davinci-001",
    "text-davinci-002",
]


def verify_openai_access(
    path_to_organization: pathlib.Path,
    path_to_api_key: pathlib.Path,
) -> None:
    """Add credentials to openai.

    Access keys stored in local files.

    Args:
        path_to_organization: path to plaintext file with organization.
        path_to_api_key: path to plaintext file with.

    Returns:
        None
    """
    openai.organization = get_plaintext_file_contents(path_to_organization)
    openai.api_key = get_plaintext_file_contents(path_to_api_key)
    return None


class OpenAIModelSettings:
    """Instance to track language model parameters before querying."""

    def __init__(
        self,
        engine: str,
        max_tokens: int = 0,
        temperature: float = 1,
        n: int = 1,  # default
        logprobs: int = 1,
        echo: bool = True,  # checking for log-probs without text completion
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        stop=None,
        params_descriptor: str = "no-complete-logprobs",
    ) -> None:
        """Set openai model settings.

        The default parameter configuration is `no-complete-logprobs` meaning that when querying a language model
        it asks to not generate any new tokens and return top-1 logprob of each token in the query.

        Args:
            engine: (OpenAI Completion API input) one of OpenAI's language model engines.
            max_tokens: (OpenAI Completion API input) max number of tokens to generate from the language model.
            temperature: (OpenAI Completion API input) randomization control for sequence generation; 1 is unconstrained, 0 is complete with most likely.
            n: (OpenAI Completion API input) number of independent completions to generate.
            logprobs: (OpenAI Completion API input) for each token, give engine's log-probabilities of this number of alternatives.
            echo: (OpenAI Completion API input) if true, give tokens and log-probabilities of query tokens.
            presence_penalty: (OpenAI Completion API input) 0 for no constraint.
            frequency_penalty: (OpenAI Completion API input) 0 for no constraint.
            stop: (OpenAI Completion API input) None for no constraint, tokens to treat as stop tokens.
            params_descriptor: descriptor for input params.

        Raises:
            ValueError: If engine is not in ENGINES.

        """
        if engine not in ENGINES:
            raise ValueError(
                f"engine {engine} is not a valid choice of OpenAI engines {*ENGINES,}"
            )

        self.engine = engine
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.n = n
        self.logprobs = logprobs
        self.echo = echo
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.stop = stop
        self.params_descriptor = params_descriptor

    def __str__(self) -> str:
        """Enables printing object as string.

        Returns:
            Dictionary of object attributes to string.
        """
        return str(vars(self))


def call_openai_api(
    prompt: str,
    model_settings: OpenAIModelSettings,
) -> dict:
    """Wrapper for getting openai response.

    Args:
        prompt: (string) query to be given to language model.
        model_settings: (OpenAI_Model_Settings) maps directly to OpenAI Completion API input, other query parameters.

    Returns:
        {
            "model": dictionary of model_settings.
            "output": dictionary of API outputs.
        }
    """
    output = openai.Completion.create(
        prompt=prompt,
        engine=model_settings.engine,
        max_tokens=model_settings.max_tokens,
        temperature=model_settings.temperature,
        n=model_settings.n,
        logprobs=model_settings.logprobs,
        echo=model_settings.echo,
        presence_penalty=model_settings.presence_penalty,
        frequency_penalty=model_settings.frequency_penalty,
        stop=model_settings.stop,
    )

    return {"model": vars(model_settings), "output": output}
