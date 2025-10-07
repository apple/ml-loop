#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

import requests

TOKEN_ID_PATTERN = re.compile(r"token_id:(\d+)")


def extract_token_id(s: str) -> int:
    # NOTE: the --return-tokens-as-token-ids option to vllm serve causes "token" field
    # to be a string of the form "token_id:3134"
    match = TOKEN_ID_PATTERN.match(s)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Invalid format: {s}")


from phi_agents.utils.logger import get_phi_logger
from phi_agents.vllm.vllm_server import LORA_MODEL_ID

if TYPE_CHECKING:
    from threading import Event

logger = get_phi_logger()


class MaxSeqLenExceeded(RuntimeError):
    """Raised when we exceed the maximum token length."""

    pass


class VLLMClient:
    """An interface to an LLM that uses a vLLM server."""

    def __init__(
        self, host: str = "localhost", port: int = 8000, cancellation_event: Event | None = None
    ):
        self._url = f"http://{host}:{port}"
        self._cancellation_event = cancellation_event

    def _get_base_model_id(self) -> str:
        """Return the base model ID."""
        response = requests.get(f"{self._url}/v1/models")
        models = response.json()["data"]
        ids: list[str] = [model["id"] for model in models]
        non_lora_ids = [id for id in ids if not id.startswith(LORA_MODEL_ID)]
        if len(non_lora_ids) != 1:
            raise ValueError(
                f"Expected exactly one model ID that does not start with {LORA_MODEL_ID}, but got models: {models}"
            )
        return non_lora_ids[0]

    def _check_for_error_and_maybe_raise(self, data: dict[str, Any]) -> None:
        if "object" in data and data["object"] == "error":
            assert "choices" not in data
            code = data["code"]
            error_type = data["type"]
            error_msg = data["message"]
            raise RuntimeError(
                f"VLLMClient with URL {self._url} received {error_type} from VLLM server with code {code}: {error_msg}"
            )

    def _is_max_tokens_stopped(self, choice: dict[str, Any]) -> bool:
        # extract reason for fnishing
        finish_reason = choice["finish_reason"]
        if finish_reason == "stop":
            return False
        elif finish_reason == "length":
            return True
        else:
            raise AssertionError(f"unexpected finish_reason: {finish_reason}")

    def _cancelled(self) -> bool:
        return self._cancellation_event is not None and self._cancellation_event.is_set()

    def get_completion(
        self,
        lora_model_id: str | None,
        prompt_tokens: list[int],
        temperature: float | None = 1.0,
        top_p: float | None = 1.0,
        top_k: int | None = -1,
        min_p: float | None = 0.0,
        max_new_tokens: int | None = 3000,
        stop_token_ids: list[int] | None = None,
        frequency_penalty: float | None = 0.0,
        repetition_penalty: float | None = 1.0,
        presence_penalty: float | None = 0.0,
        logit_bias: dict[str, float] | None = None,
        seed: int | None = None,
    ) -> tuple[str, list[int], list[float], bool, bool]:
        """Return completion of token sequence.

        Args:
            lora_model_id: ID of the requested LoRA model (the server must support it) and None if the base model should be used.
            prompt_tokens: List of prompt tokens including generation prompt (i.e. assistant message header and newlines).
            temperature: What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
            top_p: An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass.
            So 0.1 means only the tokens comprising the top 10% probability mass are considered. We generally recommend altering this or temperature but not both (from OpenAI docs).
            max_new_tokens: Maximum number of tokens to generate. If the maximum is reached, then returned tokens will not end in a stop token.
            top_k: Integer that controls the number of top tokens to consider.
            min_p: Float that represents the minimum probability for a token to be considered, relative to the probability of the most likely token. If specified, must be in [0, 1].
            stop_token_ids: Tokens to stop on.
            frequency_penalty: Float that penalizes new tokens based on their frequency in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.
            repetition_penalty: .
            presence_penalty: .
            logit_bias: supported in vllm 0.6.6 and later for special tokens like BOM.
            seed: .
        """
        model_id = lora_model_id if lora_model_id is not None else self._get_base_model_id()

        # default values disable any custom sampling logic unless specifically requested otherwise
        temperature = 1.0 if temperature is None else temperature
        top_p = 1.0 if top_p is None else top_p
        top_k = -1 if top_k is None else top_k
        min_p = 0.0 if min_p is None else min_p
        frequency_penalty = 0.0 if frequency_penalty is None else frequency_penalty
        repetition_penalty = 1.0 if repetition_penalty is None else repetition_penalty
        presence_penalty = 0.0 if presence_penalty is None else presence_penalty

        # make request to Completions API (not Chat Completions)
        headers = {"User-Agent": "VLLM Client"}
        args = {
            "model": model_id,
            "prompt": prompt_tokens,
            "logprobs": 0,  # this means provide logprobs for the chosen token but not any others,
            "skip_special_tokens": False,
            "stream": True,
        }

        if max_new_tokens is not None:
            args["max_tokens"] = max_new_tokens
        if stop_token_ids is not None:
            args["stop_token_ids"] = stop_token_ids
        if logit_bias is not None:
            args["logit_bias"] = logit_bias

        if seed is not None:
            args["seed"] = seed

        args["temperature"] = temperature
        args["top_p"] = top_p
        args["top_k"] = top_k
        args["min_p"] = min_p
        args["repetition_penalty"] = repetition_penalty
        args["frequency_penalty"] = frequency_penalty
        args["presence_penalty"] = presence_penalty

        cancelled = False
        max_tokens_stopped = False
        text_chunks: list[str] = []
        generated_tokens: list[int] = []
        generated_log_probs: list[float] = []

        # These are from the Server-Sent Events (SSE) format that both OpenAI API and vllm use, e.g. see:
        # https://platform.openai.com/docs/api-reference/runs/createThreadAndRun#runs-createthreadandrun-stream
        expected_chunk_prefix = "data: "
        final_chunk = "[DONE]"
        chunks_received = 0

        if self._cancelled():
            cancelled = True
        else:
            with requests.post(
                f"{self._url}/v1/completions", headers=headers, json=args, stream=True
            ) as r:
                try:
                    r.raise_for_status()
                except requests.HTTPError as error:
                    print(
                        f"vLLM server request failed {r.text} {r.status_code=}\n\njson_args:\n{args}"
                    )
                    json_dict = r.json()
                    if r.status_code == 400 and json_dict["message"].startswith(
                        "This model's maximum context length is "
                    ):
                        raise MaxSeqLenExceeded from error
                    else:
                        raise error

                for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                    chunks_received += 1

                    if self._cancelled():
                        print(f"VLLM rollout cancelled! {self._url}")
                        cancelled = True
                        break

                    if not chunk:
                        continue

                    if not isinstance(chunk, str) or not chunk.startswith(expected_chunk_prefix):
                        raise ValueError(
                            f"Expected data chunks in Server-Sent Events (SSE) format, got: {chunk=}, {r.status_code=} {args=}"
                        )

                    chunk = chunk[len(expected_chunk_prefix) :].strip()

                    # non-JSON final chunk signals the end of stream
                    # this is the next chunk that arrives after the chunk with "finish_reason"
                    if chunk == final_chunk:
                        if chunks_received == 1:
                            logger.error("VLLM server returned no data")
                        break

                    try:
                        data = json.loads(chunk)
                    except Exception as exc:
                        logger.error(f"Could not parse data {chunk=}, {r.status_code=} {args=}")
                        raise exc

                    if "error" in data:
                        logger.error(f"VLLM error: {data=} {chunks_received=}")
                        logger.error(f"json_args: {args}")

                    if "choices" not in data or len(data["choices"]) != 1:
                        logger.error(
                            f"VLLM error: Invalid data format (no 'choices'?) {data=} {chunks_received=}"
                        )
                        break

                    choices = data["choices"]
                    assert len(choices) == 1
                    choice = choices[0]

                    if "finish_reason" not in choice:
                        logger.error(f"VLLM error: Invalid choice format {choice=}")
                        break

                    if choice["finish_reason"] is not None:
                        max_tokens_stopped = self._is_max_tokens_stopped(choice)

                    generated_tokens.extend(
                        [extract_token_id(s) for s in choice["logprobs"]["tokens"]]
                    )
                    generated_log_probs.extend(choice["logprobs"]["token_logprobs"])
                    text_chunks.append(choice["text"])

        full_text = "".join(text_chunks)

        assert len(generated_tokens) == len(generated_log_probs)

        if not cancelled and len(generated_tokens) <= 1:
            logger.error(f"VLLM error: {len(generated_tokens)} tokens generated {args=}")
            logger.error(
                f"{generated_tokens} {generated_log_probs} {text_chunks} {chunks_received}"
            )

        return full_text, generated_tokens, generated_log_probs, max_tokens_stopped, cancelled
