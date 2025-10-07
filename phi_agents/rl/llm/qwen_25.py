#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from collections.abc import Sequence
from pathlib import Path
from threading import Event
from typing import cast

import numpy as np
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from phi_agents.rl.type_defs import (
    AssistantMessage,
    Message,
    PolicyMessage,
    PolicyTokenInfo,
    SystemMessage,
    UserMessage,
)
from phi_agents.rl.vllm_client import MaxSeqLenExceeded, VLLMClient
from phi_agents.utils.logger import get_phi_logger

from .base_llm import SpecialToken, TrainableLLM, messages_str

logger = get_phi_logger()


def contains_english_chars_only(input_str: str) -> bool:
    try:
        input_str.encode(encoding="utf-8").decode("ascii")
    except UnicodeDecodeError:
        return False
    else:
        return True


class VLLMQwen25(TrainableLLM):
    def __init__(
        self,
        host: str,
        port: int,
        base_model_path: Path,
        model_id: str | None,
        temperature: float,
        max_new_tokens: int,
        top_p: float | None,
        min_p: float | None,
        top_k: int | None,
        frequency_penalty: float | None,
        max_model_len: int,
        cancellation_event: Event | None = None,
    ):
        self._vllm = VLLMClient(host, port, cancellation_event)
        self._temperature = temperature
        self._base_model_path = base_model_path
        self._lora_model_id = model_id
        self._max_new_tokens = max_new_tokens
        self._top_p = top_p
        self._min_p = min_p
        self._top_k = top_k
        self._frequency_penalty = frequency_penalty
        self._special_tokens = {
            # There are more, but these are the common ones
            # Some tokens are noted in site below (we've seen <|endoftext|> predicted by model)
            # https://github.com/QwenLM/Qwen/blob/main/tokenization_note.md
            # Other special tokens can be found using print(tokenizer)
            "bom": SpecialToken(id=151644, content="<|im_start|>"),
            "eom": SpecialToken(id=151645, content="<|im_end|>"),
            "eot": SpecialToken(id=151643, content="<|endoftext|>"),
        }
        self._stop_tokens: set[int] = set(
            [
                self._special_tokens["eom"].id,  # '<|im_end|>'
                self._special_tokens["eot"].id,  # '<|endoftext|>'
                self._special_tokens["bom"].id,  # '<|im_start|>'
            ]
        )

        self._tokenizer = AutoTokenizer.from_pretrained(base_model_path)

        # Validation checks
        _special_token_ids = set()
        for t in self._special_tokens.values():
            # Check that decoded token matches the hardcoded content
            assert self.tokenizer.decode([t.id]) == t.content

            _special_token_ids.add(t.id)
        # Check there are no repeated tokens in special_tokens
        assert len(_special_token_ids) == len(self._special_tokens)

        # Get dummy system prompt
        self.dummy_system_msg, self.dummy_system_prompt_tokens = (
            self.get_dummy_system_prompt_tokens()
        )

        # Get generation prompt
        self.generation_prompt_tokens = self.get_generation_prompt_tokens()

        self._max_model_len = max_model_len

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    @property
    def special_tokens(self) -> dict[str, SpecialToken]:
        return self._special_tokens

    def get_dummy_system_prompt_tokens(self) -> tuple[SystemMessage, Sequence[int]]:
        dummy_msg = SystemMessage(content="dummy")
        return dummy_msg, self._tokenize_with_tokenizer(
            messages=[dummy_msg], add_generation_prompt=False
        )

    def get_generation_prompt_tokens(self) -> Sequence[int]:
        """Get generation prompt tokens (hacky)."""
        dummy_msgs = [{"role": "system", "content": "dummy"}]
        without_prompt_tokens = self.tokenizer.apply_chat_template(
            dummy_msgs,
            tokenize=True,
            add_generation_prompt=False,
        )
        with_prompt_tokens: list[int] = cast(
            list[int],
            self.tokenizer.apply_chat_template(
                dummy_msgs,
                tokenize=True,
                add_generation_prompt=True,
            ),
        )
        assert without_prompt_tokens == with_prompt_tokens[: len(without_prompt_tokens)]
        generation_prompt_tokens = with_prompt_tokens[len(without_prompt_tokens) :]
        assert generation_prompt_tokens == [151644, 77091, 198]  # '<|im_start|>assistant\n'
        return generation_prompt_tokens

    def _tokenize_with_tokenizer(
        self, messages: Sequence[Message], add_generation_prompt: bool, quiet: bool = False
    ) -> list[int]:
        """Tokenize the messages by running the tokenizer on all messages.

        NOTE: This does not guarantee that the PolicyMessage tokens match with what was generated
        since we rerun the tokenizer on the PolicyMessage content.
        """
        # self.tokenizer.apply_chat_template() has weird behavior in that:
        # - If a system msg is provided, the function works like normal.
        # - If no system msg is provided, one will be created automatically with preset content.
        # To account for the second case, we always add a dummy system msg to avoid having one
        # being automatically generated.
        if not isinstance(messages[0], SystemMessage):
            _messages: Sequence[Message] = [self.dummy_system_msg] + list(messages)
        else:
            _messages = messages

        messages_for_tokenizer: list[dict[str, str]] = []
        for m in _messages:
            match m:
                case AssistantMessage() | PolicyMessage():
                    if not quiet and isinstance(m, PolicyMessage):
                        logger.warning(
                            "Using tokenizer to retokenize PolicyMessage instead of using the "
                            "actual generated tokens."
                        )
                    m_for_tokenizer = {"role": "assistant", "content": m.content}
                case SystemMessage():
                    m_for_tokenizer = {"role": "system", "content": m.content}
                case UserMessage():
                    m_for_tokenizer = {"role": "user", "content": m.content}
                case _:
                    raise ValueError("Unsupported message type.")
            messages_for_tokenizer.append(m_for_tokenizer)
        tokens = self.tokenizer.apply_chat_template(
            messages_for_tokenizer,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
        assert isinstance(tokens, list) and isinstance(tokens[0], int)

        # Remove the dummy system msg tokens if the dummy system msg was added
        if not isinstance(messages[0], SystemMessage):
            n_sys_prompt_tokens = len(self.dummy_system_prompt_tokens)
            assert tokens[:n_sys_prompt_tokens] == self.dummy_system_prompt_tokens
            tokens = tokens[n_sys_prompt_tokens:]

        return cast(list[int], tokens)

    def get_tokens(
        self, messages: list[Message], is_output: bool = False, log_probs: bool = False
    ) -> tuple[list[int], list[bool] | None, list[float] | None]:
        """Tokenize messages.

        Args:
            messages: .
            is_output: If True, return a list of bools of len(tokens). Else return None.
            log_probs: If True, return the log probs for the tokens belonging to policymessages.
                Else return None.
        """
        # Minor optimization: tokenize all msgs up to the first policymessage
        msg_idx = 0
        while msg_idx < len(messages) and not isinstance(messages[msg_idx], PolicyMessage):
            msg_idx += 1
        running_tokens_list = self._tokenize_with_tokenizer(
            messages[:msg_idx], add_generation_prompt=False
        )
        is_output_list: list[bool] | None = (
            [False] * len(running_tokens_list) if is_output else None
        )
        log_probs_list: list[float] | None = (
            [np.nan] * len(running_tokens_list) if log_probs else None
        )

        # Tokenize message by message
        newline_tokens: list[int] = self.tokenizer.encode("\n")  # Should be [198]
        for cur_msg_idx in range(msg_idx, len(messages)):
            cur_msg: Message = messages[cur_msg_idx]
            match cur_msg:
                case PolicyMessage():
                    # Reminder: A tokenized msg looks like:
                    # <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
                    # Policy msg generated tokens will include <|im_end|> but not \n
                    # Thus we need to manually add \n
                    assert cur_msg.prompt_tokens == self.generation_prompt_tokens

                    maybe_stop_token: list[int] = []
                    if len(cur_msg.generated_tokens) > 0:
                        if cur_msg.stopped_by_max_tokens_limit:
                            if cur_msg.generated_tokens[-1] not in self._stop_tokens:
                                maybe_stop_token = [self._special_tokens["eom"].id]
                            else:
                                logger.error(
                                    f"Stopped by max tokens but last tokens: {cur_msg.generated_tokens[-5:]} {self._stop_tokens=}"
                                )
                        else:
                            assert cur_msg.generated_tokens[-1] in self._stop_tokens
                    else:
                        logger.error(
                            f"Policy message {cur_msg_idx} has no generated tokens {cur_msg=}"
                        )
                        maybe_stop_token = [self._special_tokens["eom"].id]

                    # Tokenized message = bom tokens + msg tokens + stop token + eom tokens
                    tokenized_cur_msg = (
                        cur_msg.prompt_tokens
                        + cur_msg.generated_tokens
                        + maybe_stop_token
                        + newline_tokens
                    )
                    is_output_cur_msg = (
                        [False] * len(cur_msg.prompt_tokens)
                        + [True] * len(cur_msg.generated_tokens)
                        + [False] * len(maybe_stop_token)
                        + [False] * len(newline_tokens)
                    )
                    log_probs_cur_msg = (
                        [np.nan] * len(cur_msg.prompt_tokens)
                        + cur_msg.generated_token_logprobs
                        + [np.nan] * len(maybe_stop_token)
                        + [np.nan] * len(newline_tokens)
                    )

                    running_tokens_list.extend(tokenized_cur_msg)
                    if is_output_list is not None:
                        is_output_list.extend(is_output_cur_msg)
                    if log_probs_list is not None:
                        log_probs_list.extend(log_probs_cur_msg)

                    # Sanity check (only check up to :-2 because sometimes the agent predicts
                    # <|endoftext|> instead of <|im_end|>)
                    reference_msg_tokens = self._tokenize_with_tokenizer(
                        [cur_msg], add_generation_prompt=False, quiet=True
                    )
                    reference_msg_txt = self.tokenizer.decode(reference_msg_tokens[:-2])
                    cur_msg_txt = self.tokenizer.decode(tokenized_cur_msg[:-2])
                    if contains_english_chars_only(reference_msg_txt + cur_msg_txt):
                        # We run assertion only if we have no non-english chars
                        # Otherwise this assertion fails on non-english chars e.g. special chars
                        assert reference_msg_txt == cur_msg_txt, (
                            "Mismatch in manual vs chat tokens:\n"
                            "Tokenized chat version:\n"
                            f"{self.tokenizer.decode(reference_msg_tokens)}\n"
                            "Tokenized msg version:\n"
                            f"{self.tokenizer.decode(tokenized_cur_msg)}\n"
                            "Tokens:\n"
                            f"First tokens: {reference_msg_tokens[:10]} vs {tokenized_cur_msg[:10]}\n"
                            f"Last tokens: {reference_msg_tokens[-10:]} vs {tokenized_cur_msg[-10:]}\n"
                            f"Lengths: {len(reference_msg_tokens)} vs {len(tokenized_cur_msg)}\n"
                            f"cur msg idx: {cur_msg_idx}\n"
                            f"len(messages): {len(messages)}\n"
                            f"{self.tokenizer.decode(reference_msg_tokens[-2:])}\n"
                            "-------------\n"
                            f"{self.tokenizer.decode(tokenized_cur_msg[-2:])}\n"
                            "End\n"
                        )
                case _:
                    cur_msg_tokens = self._tokenize_with_tokenizer(
                        [cur_msg], add_generation_prompt=False
                    )
                    running_tokens_list.extend(cur_msg_tokens)
                    if is_output_list is not None:
                        is_output_list.extend([False] * len(cur_msg_tokens))
                    if log_probs_list is not None:
                        log_probs_list.extend([np.nan] * len(cur_msg_tokens))

        # More checks
        if is_output_list is not None:
            assert len(running_tokens_list) == len(is_output_list)
        if log_probs_list is not None:
            assert len(running_tokens_list) == len(log_probs_list)

        return running_tokens_list, is_output_list, log_probs_list

    def generate(self, messages: list[Message]) -> PolicyMessage:
        r"""Generate a (policy) message response to messages.

        Tokenization note:
            Using apply_chat_template(..., add_generation_prompt=False) gives something like:
                <|im_start|>user\n<message_content><|im_end|>\n

            Using apply_chat_template(..., add_generation_prompt=True) gives something like:
                <|im_start|>user\n<message_content><|im_end|>\n<|im_start|>assistant\n

            However using vllm generate completion gives something like:
                <|im_start|>user\n<message_content><|im_end|>

            So there is no \n token in the output of generate(). This means we should not generate
            on manually concatenated tokens (e.g. prompt_tokens + generate().generated_tokens).
            Instead, use self.get_tokens().
        """
        # Similar to self._tokenize_with_tokenizer(messages, add_generation_prompt=True)
        # but re-use generated tokens instead of retokenizing
        prompt_tokens = self.get_tokens(messages)[0] + list(self.generation_prompt_tokens)

        n_prompt_tokens = len(prompt_tokens)
        if n_prompt_tokens + self._max_new_tokens >= self._max_model_len:
            max_new_tokens = self._max_model_len - n_prompt_tokens - 1
            logger.info(f"Adjusting {max_new_tokens=} (default: {self._max_new_tokens})")
        else:
            max_new_tokens = self._max_new_tokens

        if max_new_tokens <= 1:
            raise MaxSeqLenExceeded(
                f"Ran out of context length: {self._max_model_len=} {n_prompt_tokens=} {self._max_new_tokens=} {max_new_tokens=}"
            )

        _, generated_tokens, log_probs, max_tokens_stopped, cancelled = self._vllm.get_completion(
            self._lora_model_id,
            prompt_tokens,
            stop_token_ids=list(self._stop_tokens),
            temperature=self._temperature,
            top_p=self._top_p,
            min_p=self._min_p,
            top_k=self._top_k,
            frequency_penalty=self._frequency_penalty,
            max_new_tokens=max_new_tokens,
        )
        text_tokens = []
        text_content = ""

        if len(generated_tokens) > 0:
            text_tokens = generated_tokens

            def _diag_info() -> str:
                tail = generated_tokens[-10:]
                return f"{len(generated_tokens)} {tail=} {cancelled=} {self._stop_tokens=} {self.tokenizer.decode(tail)=}"

            if max_tokens_stopped and generated_tokens[-1] in self._stop_tokens:
                logger.warning(
                    f"Stopped due to max_tokens but last token is a stop token! {_diag_info()}"
                )
                text_tokens = generated_tokens[:-1]
            elif not max_tokens_stopped and not cancelled:
                assert (
                    generated_tokens[-1] in self._stop_tokens
                ), f"Last token was not in stop tokens: {_diag_info()}"
                text_tokens = generated_tokens[:-1]

            text_content = self.tokenizer.decode(text_tokens)
        elif not cancelled:
            logger.error(
                f"No tokens generated! {n_prompt_tokens=} {max_new_tokens=} {messages_str(messages)=}"
            )

        return PolicyMessage(
            text_content,
            stopped_by_max_tokens_limit=max_tokens_stopped,
            prompt_tokens=list(self.generation_prompt_tokens),
            generated_tokens=generated_tokens,
            generated_token_logprobs=log_probs,
            ipython=False,
            rollout_cancelled=cancelled,
        )

    def get_policy_token_info(self, messages: list[Message]) -> PolicyTokenInfo:
        tokens, is_output, log_probs = self.get_tokens(messages, is_output=True, log_probs=True)
        assert is_output is not None and log_probs is not None
        return PolicyTokenInfo(tokens, log_probs, is_output)
