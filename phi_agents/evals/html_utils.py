#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import html
from typing import Any

import wandb
import wandb.plot
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from phi_agents.appworld.interface import Failure, Pass
from phi_agents.evals.appworld_evals import Episode, TaskEvalResult
from phi_agents.evals.appworld_rollout_data import AppWorldTrainingRollout
from phi_agents.rl.type_defs import Message
from phi_agents.utils.logger import get_phi_logger

logger = get_phi_logger()


def rollout_summary_html(r: AppWorldTrainingRollout) -> str:
    data = r.appworld_rollout_data
    return f"""
    <div class="eval-result">
    dataset: {data.dataset_name}, task: {data.task.task_id}, diff: {data.eval_result.difficulty},
    success: {data.eval_result.success}, tests: {len(data.eval_result.passes)}/{data.eval_result.num_tests}
    </div>
    """


def get_message_html(
    msg: Message, idx: int, num_message_tokens: int, total_tokens: int, interaction: int
) -> str:
    return f"""
    <div class="chat-message {msg.role}">
    <pre><p><b>{idx}: {msg.role}</b>    message tokens {num_message_tokens}    total tokens {total_tokens}    interaction {interaction}</pre></p>
    <pre class="chat-message-content">{html.escape(msg.content)}</pre>
    </div>
    """


def get_messages_html(
    episode: Episode, tokenizer: PreTrainedTokenizerBase, skip_system_prompt: bool
) -> str:
    from phi_agents.visualization.rollouts import get_approx_num_tokens_in_message

    assert episode.num_prompt_messages is not None
    total_tokens = 0
    message_htmls = []
    interaction_count = 0
    for idx, msg in enumerate(episode.chat_history):
        if idx >= episode.num_prompt_messages:
            interaction_count += 1

        num_message_tokens = get_approx_num_tokens_in_message(tokenizer, msg)
        total_tokens += num_message_tokens

        if skip_system_prompt and idx < episode.num_prompt_messages - 1:
            continue

        message_htmls.append(
            get_message_html(
                msg,
                idx,
                num_message_tokens=num_message_tokens,
                total_tokens=total_tokens,
                interaction=(interaction_count + 1) // 2,
            )
        )

    message_html_str = "\n".join(message_htmls)

    html_content = f"""
    <div>
        <div class="chat-container">
        {message_html_str}
        </div>
    </div>
    """

    return html_content


def get_test_html(test: Pass | Failure) -> str:
    return (
        f"<div>"
        f"<div>{test.requirement}</div>"
        f"<div>{test.label}</div>"
        f"<div>{test.trace if hasattr(test, 'trace') else ''}</div>"
        "</div>"
    )


def get_tests_html(tests: list[Any]) -> str:
    return f"<div>{'\n'.join([get_test_html(test) for test in tests])}</div>"


def get_eval_html(task_eval_result: TaskEvalResult) -> str:
    return (
        f'<div class="eval-result">'
        f"<p><b>Evaluation</b></p>"
        f"<p>success: {task_eval_result.success}</p>"
        f"<p>difficulty: {task_eval_result.difficulty}</p>"
        f"<p>passed tests: {len(task_eval_result.passes)}/{task_eval_result.num_tests}</p>"
        f"<div><p><b>Passed Tests</b></p>{get_tests_html(task_eval_result.passes)}</div>"
        f"<div><p><b>Failed Tests</b></p>{get_tests_html(task_eval_result.failures)}</div>"
        f"<div></div>"
        "</div>"
    )


def rollout_html_wandb(
    r: AppWorldTrainingRollout, tokenizer: PreTrainedTokenizerBase
) -> wandb.Html:
    css_styles_compact = """
    <style>
    .eval-result {
        padding: 6px;
        border-radius: 10px;
        background-color: #dddddd;
        width: 500px;
        font-family: Arial, sans-serif;
        font-size: 10px;
    }
    .chat-message-content {
        white-space: pre-wrap; /* Preserve newlines and wrap lines */
        word-wrap: break-word; /* Break long words */
        overflow-wrap: break-word; /* Ensure wrapping in modern browsers */
    }
    .chat-message {
        padding: 6px;
        border-radius: 10px;
        background-color: #f1f1f1;
        width: 500px;
        font-family: Arial, sans-serif;
        font-size: 10px;
    }
    .chat-message.system {
        background-color: #d3e5ff;
        align-self: flex-start;
        text-align: left;
    }
    .chat-message.user {
        background-color: #d3ffd3;
        align-self: flex-start;
        text-align: left;
    }
    .chat-message.ipython {
        background-color: #fff2d3;
        align-self: flex-start;
        text-align: left;
    }
    .chat-message.assistant {
        background-color: #ffd3d3;
        align-self: flex-start;
        text-align: left;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 2px;
    }
    .chat-container p,
    .chat-container pre {
        margin: 0.5em 0;
        padding: 0;
    }
    </style>
    """

    from phi_agents.visualization.rollouts import convert_rollout_to_episode

    episode = convert_rollout_to_episode(r)
    summary_html = rollout_summary_html(r)
    messages_html = get_messages_html(episode, tokenizer, skip_system_prompt=True)
    eval_html = get_eval_html(r.appworld_rollout_data.eval_result)

    rollout_html = f"{summary_html}{messages_html}{eval_html}"

    html = f"""
    <html>
        <head>{css_styles_compact}</head>
        <body>
        <p>{rollout_html}</p>
        </body>
    </html>
    """

    return wandb.Html(html, inject=False)
