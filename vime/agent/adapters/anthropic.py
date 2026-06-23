"""Anthropic Messages adapter for agent rollouts.

<<<<<<< /home/aoshen/vime/projects/slime-sync-2118/agent_run/results/build_3way/tmp_ours.txt
The adapter exposes ``/v1/messages`` and ``/v1/messages/count_tokens``. It
renders each Anthropic message history with the served model's chat template,
calls vLLM's ``/inference/v1/generate`` with ``token_ids``, and
records the exact sampled token ids/logprobs as ``TurnRecord`` objects. New
code should use ``AnthropicAdapter`` and call ``finish_session()`` at trajectory
end to drain trainable ``TokenSegment`` objects.

It also handles Claude Code sub-agent and compaction patterns by splitting one
session into ``subagent``, ``wipe``, and ``final`` segments.
=======
Exposes /v1/messages and /v1/messages/count_tokens. Each Anthropic message
history is rendered with the served model's chat template, sent to vllm
/generate as input_ids, and fed into a shared TrajectoryManager keyed by session
id. finish_session(sid) drains a session's trajectory into a list of Sample.

The per-sid tree inside TrajectoryManager handles sub-agent and compaction
patterns automatically: any divergence in the prompt prefix forks into a new
leaf, so we do not track explicit chains here.

This module mirrors slime.agent.adapters.openai; the section layout (adapter
class -> translation -> reply building -> request framing) is shared between
them. See BaseAdapter for the hooks to fill.
>>>>>>> /home/aoshen/vime/projects/slime-sync-2118/agent_run/results/build_3way/tmp_theirs.txt
"""

from __future__ import annotations

import json
import logging
import secrets
from typing import Any

from aiohttp import web

<<<<<<< /home/aoshen/vime/projects/slime-sync-2118/agent_run/results/build_3way/tmp_ours.txt
from vime.agent.adapters.common import ADAPTER_KEY, REASONING_PARSER_KEY, TOKENIZER_KEY, TOOL_PARSER_KEY
from vime.agent.adapters.common import AdapterChain as Chain
from vime.agent.adapters.common import (
=======
from slime.agent.adapters.common import (
>>>>>>> /home/aoshen/vime/projects/slime-sync-2118/agent_run/results/build_3way/tmp_theirs.txt
    BaseAdapter,
    Reply,
    flatten_content,
    manager_finish_reason,
    sid_from_bearer,
    tool_call_dict,
)
<<<<<<< /home/aoshen/vime/projects/slime-sync-2118/agent_run/results/build_3way/tmp_ours.txt
from vime.agent.adapters.common import stable_hash as _hash
from vime.agent.parsing import parse_model_output
from vime.agent.trajectory import TokenSegment, TurnRecord, TurnSegment, make_turn_segment, merge_turn_segments
=======
from slime.agent.parsing import ParsedModelOutput
>>>>>>> /home/aoshen/vime/projects/slime-sync-2118/agent_run/results/build_3way/tmp_theirs.txt

logger = logging.getLogger(__name__)


class AnthropicAdapter(BaseAdapter):
    """Anthropic Messages-compatible HTTP adapter: wire translation and reply
    framing only; the turn machinery is inherited from BaseAdapter."""

    logger = logger
    log_prefix = "anthropic_adapter"
    max_token_keys = ("max_tokens",)
    stop_keys = ("stop_sequences",)

    def _register_routes(self, app: web.Application) -> None:
        app.router.add_post("/v1/messages", self._run_turn)
        app.router.add_post("/v1/messages/count_tokens", _count_tokens)

    def _session_id(self, request: web.Request, body: dict) -> str:
        return _request_session_id(request)

    def _preprocess_body(self, body: dict) -> None:
        _fold_mid_list_system_into_user(body)

    def _translate(self, body: dict) -> tuple[list[dict], list[dict] | None]:
        translated = _translate_messages(body.get("messages") or [], body.get("system"))
        tools_schema = _tools_to_chat_tools(body.get("tools"))
        return translated, tools_schema

    def _build_reply(self, parsed, raw_finish, translated, tools_schema) -> Reply:
        blocks, stop_reason, manager_message = _build_reply_parts(parsed, raw_finish)
        return Reply(
            manager_message=manager_message,
            finish_reason=manager_finish_reason(parsed.tool_uses, raw_finish),
            wire=(blocks, stop_reason),
        )

    async def _respond(self, request, body, reply, in_tok, out_tok, stream) -> web.StreamResponse:
        blocks, stop_reason = reply.wire
        if stream:
            return await _render_stream(request, blocks, stop_reason, in_tok, out_tok)
        return web.json_response(_render_response(body, blocks, stop_reason, in_tok, out_tok))


# --- Translation (Anthropic wire -> chat-template messages) ---


def _translate_messages(msgs: list[dict], system: Any) -> list[dict]:
    """Anthropic messages + system -> chat-template messages. Pure function."""
    translated: list[dict] = []
    if system:
        translated.append({"role": "system", "content": flatten_content(system)})
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role, content = m.get("role"), m.get("content")
        if role == "user":
            blocks = content if isinstance(content, list) else [{"type": "text", "text": flatten_content(content)}]
            for b in blocks:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    translated.append({"role": "tool", "content": flatten_content(b.get("content"))})
                elif isinstance(b, dict) and b.get("type") == "text":
                    translated.append({"role": "user", "content": b.get("text", "")})
                else:
                    translated.append({"role": "user", "content": flatten_content(b)})
        elif role == "assistant":
            texts, thinkings, tcs = [], [], []
            blocks = content if isinstance(content, list) else [{"type": "text", "text": flatten_content(content)}]
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text":
                    texts.append(b.get("text", ""))
                elif b.get("type") == "thinking":
                    thinkings.append(b.get("thinking", ""))
                elif b.get("type") == "tool_use":
                    # drop the wire-only id; tool_call_dict keeps arguments a dict
                    tcs.append(tool_call_dict(b.get("name", "tool"), b.get("input")))
            mo: dict[str, Any] = {"role": "assistant", "content": "".join(texts)}
            if thinkings:
                mo["reasoning_content"] = "".join(thinkings)
            if tcs:
                mo["tool_calls"] = tcs
            translated.append(mo)
        elif role == "system":
            translated.append({"role": "system", "content": flatten_content(content)})
    return translated


def _tools_to_chat_tools(anth_tools: list[dict] | None) -> list[dict] | None:
    """Convert Anthropic tools to tokenizer chat-template tool schema."""
    if not anth_tools:
        return None
    ts: list[dict] = []
    for t in anth_tools:
        if not isinstance(t, dict) or "name" not in t:
            continue
        ts.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema") or t.get("parameters") or {"type": "object", "properties": {}},
                },
            }
        )
    return ts or None


<<<<<<< /home/aoshen/vime/projects/slime-sync-2118/agent_run/results/build_3way/tmp_ours.txt
def _replace_chat_messages(target: Chain, body: dict) -> None:
    """new/wipe: full reset of chat state and turn log."""
    all_msgs = body.get("messages") or []
    target.chat_messages = _translate_anthropic(all_msgs, body.get("system"))
    if "system" in body:
        target.system_hash = _hash(body.get("system"))
    target.turns.clear()
    target.seen_msgs = len(all_msgs)
    target.msg_hashes = [_hash(m) for m in all_msgs]
    if target.tools_schema is None:
        target.tools_schema = _anthropic_tools_to_chat_tools(body.get("tools"))


def _extend_chat_messages(target: Chain, body: dict) -> None:
    """append: translate only the new tail."""
    all_msgs = body.get("messages") or []
    translated = _translate_anthropic(all_msgs[target.seen_msgs :], None)
    target.chat_messages.extend(translated)

    target.seen_msgs = len(all_msgs)
    target.msg_hashes = [_hash(m) for m in all_msgs]
    if target.tools_schema is None:
        target.tools_schema = _anthropic_tools_to_chat_tools(body.get("tools"))


def _build_prompt(target: Chain, body: dict, kind: str, tok) -> list[int]:
    """Replace/extend chat_messages and render input ids for vLLM."""
    (_extend_chat_messages if kind == "append" else _replace_chat_messages)(target, body)
    return render_token_ids(target, tok)


async def _generate(
    prompt_ids: list[int], s: Session, body: dict, app, *, session_id: str | None = None
) -> TurnRecord:
    """Call vLLM and return a TurnRecord.

    1. build sampling_params (session defaults overlaid with body overrides)
    2. POST vLLM ``/inference/v1/generate``; on cancel/error tear down
       the request (vLLM has no per-request HTTP abort endpoint)
    3. keep the exact prompt/output token ids; trajectory merge later compares
       later prompt tokens with earlier outputs to build the loss mask
    """
    return await call_vllm_generate(
        prompt_ids,
        s,
        body,
        app,
        max_token_keys=("max_tokens",),
        stop_keys=("stop_sequences",),
        log_prefix="anthropic_adapter",
        logger=logger,
        session_id=session_id,
    )
=======
# --- Reply building: parsed output -> Anthropic blocks + manager_message ---

>>>>>>> /home/aoshen/vime/projects/slime-sync-2118/agent_run/results/build_3way/tmp_theirs.txt

def _build_reply_parts(
    parsed: ParsedModelOutput,
    finish: str,
) -> tuple[list[dict], str, dict[str, Any]]:
    """Return (anthropic blocks, wire stop_reason, manager_message).

<<<<<<< /home/aoshen/vime/projects/slime-sync-2118/agent_run/results/build_3way/tmp_ours.txt
def _build_reply(target: Chain, output_ids: list[int], finish: str, app) -> tuple[list[dict], str, str]:
    """Turn the model's raw output ids into the reply we send back to claude-code.

    1. parse decoded text -> (thinking, visible, tool_uses) via parsers
    2. pack into Anthropic content blocks; tag dispatch_id when a tool_use
       names Task/Agent (sub-agent trigger)
    3. derive stop_reason: 'tool_use' | 'max_tokens' | 'end_turn'

    Returns (blocks, stop_reason, dispatch_id).
    """
    tok = app[TOKENIZER_KEY]

    raw_output = tok.decode(output_ids, skip_special_tokens=False) if output_ids else ""
    parsed = parse_model_output(
        raw_output,
        tokenizer=tok,
        tools_schema=target.tools_schema,
        tool_parser_name=app[TOOL_PARSER_KEY],
        reasoning_parser_name=app[REASONING_PARSER_KEY],
    )
    blocks, dispatch_id = _anthropic_blocks(parsed.reasoning, parsed.text, parsed.tool_uses)
    return blocks, _stop_reason(parsed.tool_uses, finish), dispatch_id


def _anthropic_blocks(thinking: str, visible: str, tool_uses: list[dict]) -> tuple[list[dict], str]:
    """Pack parsed model output into Anthropic content blocks."""
=======
    The tool_calls inside manager_message use canonical args (tool_call_dict) so
    this assistant turn compares equal (dict equality) to the same turn replayed
    as history on the next request.
    """
>>>>>>> /home/aoshen/vime/projects/slime-sync-2118/agent_run/results/build_3way/tmp_theirs.txt
    blocks: list[dict] = []
    if parsed.reasoning:
        blocks.append({"type": "thinking", "thinking": parsed.reasoning})
    if parsed.text:
        blocks.append({"type": "text", "text": parsed.text})

    manager_tcs: list[dict] = []
    for tu in parsed.tool_uses:
        tu_id = f"toolu_{secrets.token_hex(8)}"
        blocks.append({"type": "tool_use", "id": tu_id, "name": tu["name"], "input": tu["input"]})
        # tu_id is wire-only; tool_call_dict drops it so the leaf matches its echo
        manager_tcs.append(tool_call_dict(tu["name"], tu.get("input")))

    if not blocks:
        blocks.append({"type": "text", "text": ""})

    if parsed.tool_uses:
        stop_reason = "tool_use"
    elif finish == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"

    manager_message: dict[str, Any] = {"role": "assistant", "content": parsed.text or ""}
    if parsed.reasoning:
        manager_message["reasoning_content"] = parsed.reasoning
    if manager_tcs:
        manager_message["tool_calls"] = manager_tcs

    return blocks, stop_reason, manager_message


# --- Request framing: session id + wire response/stream rendering ---


def _request_session_id(request: web.Request) -> str:
    # Anthropic auth lands in Authorization: Bearer or X-Api-Key; the Messages
    # body carries no sid hint. Bearer wins when both are present.
    return sid_from_bearer(request) or (request.headers.get("X-Api-Key") or "").strip() or "default"


def _render_response(body: dict, blocks: list[dict], stop_reason: str, in_tok: int, out_tok: int) -> dict:
    return {
        "id": f"msg_{secrets.token_hex(12)}",
        "type": "message",
        "role": "assistant",
        "model": body.get("model", "vime-actor"),
        "content": blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
    }


async def _render_stream(request, blocks, stop_reason, in_tok, out_tok) -> web.StreamResponse:
    """Stream blocks back as an Anthropic Messages SSE response: message_start,
    (content_block_start, content_block_delta, content_block_stop)*N,
    message_delta, message_stop."""
    out = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await out.prepare(request)

    ms_data = {
        "type": "message_start",
        "message": {
            "id": f"msg_{secrets.token_hex(12)}",
            "type": "message",
            "role": "assistant",
            "model": "vime-actor",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": in_tok, "output_tokens": 0},
        },
    }
    await out.write(f"event: message_start\ndata: {json.dumps(ms_data, ensure_ascii=False)}\n\n".encode())

    for idx, block in enumerate(blocks):
        bt = block["type"]
        if bt == "thinking":
            start = {"type": "thinking", "thinking": ""}
            delta = {"type": "thinking_delta", "thinking": block["thinking"]}
        elif bt == "text":
            start = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block["text"]}
        else:  # tool_use
            start = {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}
            delta = {
                "type": "input_json_delta",
                "partial_json": json.dumps(block["input"], ensure_ascii=False),
            }

        cbs_data = {"type": "content_block_start", "index": idx, "content_block": start}
        await out.write(f"event: content_block_start\ndata: {json.dumps(cbs_data, ensure_ascii=False)}\n\n".encode())

        cbd_data = {"type": "content_block_delta", "index": idx, "delta": delta}
        await out.write(f"event: content_block_delta\ndata: {json.dumps(cbd_data, ensure_ascii=False)}\n\n".encode())

        cbe_data = {"type": "content_block_stop", "index": idx}
        await out.write(f"event: content_block_stop\ndata: {json.dumps(cbe_data, ensure_ascii=False)}\n\n".encode())

    md_data = {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
    }
    await out.write(f"event: message_delta\ndata: {json.dumps(md_data, ensure_ascii=False)}\n\n".encode())

    mst_data = {"type": "message_stop"}
    await out.write(f"event: message_stop\ndata: {json.dumps(mst_data, ensure_ascii=False)}\n\n".encode())

    return out


# count_tokens runs every turn but the client uses it only as a hint, not a
# hard budget, so returning 0 is fine.
async def _count_tokens(request: web.Request) -> web.Response:
    await request.read()
    return web.json_response({"input_tokens": 0})


# --- Anthropic-specific quirks: mid-list system folding ---


_MID_SYSTEM_WRAP_PREFIX = "<system-reminder>\n"
_MID_SYSTEM_WRAP_SUFFIX = "\n</system-reminder>\n"


def _fold_mid_list_system_into_user(body_obj: dict) -> bool:
    """Fold non-leading role:system messages into a neighbouring user message as
    a <system-reminder> text block. Mutates body_obj in place; returns True iff
    any fold happened.

    Some clients insert a system message in the middle of the message list, but
    many chat templates reject any system message past index 0. Attaching the
    wrapped reminder to the preceding user message (or the next one, if there is
    no prior user message) keeps the history acceptable to the template.
    """
    msgs = body_obj.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return False

    system_idx = [i for i, m in enumerate(msgs) if isinstance(m, dict) and m.get("role") == "system" and i > 0]
    if not system_idx:
        return False

    def _promote_to_list(msg: dict) -> list:
        c = msg.get("content")
        if isinstance(c, list):
            return c
        msg["content"] = [{"type": "text", "text": c if isinstance(c, str) else ""}]
        return msg["content"]

    def _wrap(text: str) -> dict:
        return {
            "type": "text",
            "text": _MID_SYSTEM_WRAP_PREFIX + text + _MID_SYSTEM_WRAP_SUFFIX,
        }

    changed = False
    TOMBSTONE: dict = {"__folded__": True}
    for i in system_idx:
        sys_msg = msgs[i]
        wrapped = _wrap(flatten_content(sys_msg.get("content")))
        target = None
        for j in range(i - 1, -1, -1):
            cand = msgs[j]
            if isinstance(cand, dict) and cand.get("role") == "user":
                target = cand
                _promote_to_list(target).append(wrapped)
                break
        if target is None:
            for j in range(i + 1, len(msgs)):
                cand = msgs[j]
                if isinstance(cand, dict) and cand.get("role") == "user":
                    target = cand
                    _promote_to_list(target).insert(0, wrapped)
                    break
        if target is None:
            msgs[i] = {"role": "user", "content": [wrapped]}
            changed = True
            continue
        msgs[i] = TOMBSTONE
        changed = True

    if changed:
        body_obj["messages"] = [m for m in msgs if m is not TOMBSTONE]
    return changed
