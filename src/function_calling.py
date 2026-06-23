"""
Prompt-based OpenAI function/tool calling support for the Claude Code wrapper.

The Claude Agent SDK used by this wrapper does not expose native OpenAI-style
function-calling passthrough. This module implements a *prompt-based*
(best-effort) approach instead:

  1. The caller's OpenAI ``tools`` (or legacy ``functions``) JSON schemas are
     rendered into a system-prompt fragment that instructs Claude when and how
     to emit a tool call as a single fenced ``json`` block containing a
     well-defined envelope.
  2. Claude's raw text response is then parsed: if the envelope is present, the
     tool calls are extracted and re-shaped into the OpenAI ``tool_calls``
     response format (with generated ids and JSON-string arguments). Otherwise
     the text is treated as a normal assistant message.

This is intentionally honest about being best-effort: model output is not
guaranteed to follow the envelope perfectly, and parsing is defensive.
"""

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# Sentinel marker used inside the fenced JSON envelope. The presence of the
# ``tool_calls`` key is what we look for when parsing Claude's response.
ENVELOPE_KEY = "tool_calls"


def normalize_legacy_functions(
    functions: Optional[List[Dict[str, Any]]],
    function_call: Optional[Union[str, Dict[str, Any]]],
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Union[str, Dict[str, Any]]]]:
    """
    Normalize the legacy OpenAI ``functions``/``function_call`` request fields
    into the modern ``tools``/``tool_choice`` form.

    Args:
        functions: Legacy list of function definitions, each a dict with
            ``name``/``description``/``parameters`` keys.
        function_call: Legacy function-call control, either a string
            ("auto"/"none") or a dict like ``{"name": "..."}``.

    Returns:
        A ``(tools, tool_choice)`` tuple in the modern format. Either element
        may be ``None`` if the corresponding legacy field was absent.
    """
    tools: Optional[List[Dict[str, Any]]] = None
    if functions:
        tools = [{"type": "function", "function": fn} for fn in functions]

    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    if function_call is not None:
        if isinstance(function_call, str):
            # "auto"/"none" map straight through; OpenAI also allowed these.
            tool_choice = function_call
        elif isinstance(function_call, dict) and function_call.get("name"):
            tool_choice = {
                "type": "function",
                "function": {"name": function_call["name"]},
            }

    return tools, tool_choice


def resolve_tools(
    tools: Optional[List[Dict[str, Any]]],
    tool_choice: Optional[Union[str, Dict[str, Any]]],
    functions: Optional[List[Dict[str, Any]]] = None,
    function_call: Optional[Union[str, Dict[str, Any]]] = None,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Union[str, Dict[str, Any]]]]:
    """
    Resolve the effective ``(tools, tool_choice)`` from both the modern and
    legacy request fields. Modern fields take precedence; legacy fields are
    normalized and used only as a fallback.
    """
    legacy_tools, legacy_choice = normalize_legacy_functions(functions, function_call)

    effective_tools = tools if tools else legacy_tools
    effective_choice = tool_choice if tool_choice is not None else legacy_choice

    return effective_tools, effective_choice


def _format_tool(tool: Dict[str, Any]) -> Optional[str]:
    """Render a single OpenAI tool definition into a prompt description."""
    fn = tool.get("function") if isinstance(tool, dict) else None
    if not isinstance(fn, dict):
        return None

    name = fn.get("name")
    if not name:
        return None

    description = fn.get("description", "")
    parameters = fn.get("parameters", {})

    try:
        schema_str = json.dumps(parameters, indent=2)
    except (TypeError, ValueError):
        schema_str = "{}"

    lines = [f"- {name}: {description}".rstrip()]
    lines.append(f"  parameters (JSON schema): {schema_str}")
    return "\n".join(lines)


def build_tool_prompt(
    tools: Optional[List[Dict[str, Any]]],
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
) -> Optional[str]:
    """
    Build the system-prompt fragment instructing Claude how and when to call a
    tool, given a list of OpenAI tool definitions.

    Args:
        tools: List of OpenAI tool definitions
            (``{"type": "function", "function": {...}}``).
        tool_choice: Optional OpenAI ``tool_choice`` controlling whether/which
            tool must be called ("auto"/"none"/"required" or a specific
            ``{"type": "function", "function": {"name": ...}}``).

    Returns:
        The system-prompt fragment, or ``None`` when no usable tools are
        provided or ``tool_choice`` is "none".
    """
    if not tools:
        return None

    # tool_choice == "none" disables tool calling entirely.
    if isinstance(tool_choice, str) and tool_choice == "none":
        return None

    formatted = [t for t in (_format_tool(tool) for tool in tools) if t]
    if not formatted:
        return None

    tool_descriptions = "\n".join(formatted)

    # Determine the calling directive based on tool_choice.
    forced_name: Optional[str] = None
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function")
        if isinstance(fn, dict):
            forced_name = fn.get("name")

    if forced_name:
        directive = (
            f"You MUST call the function named \"{forced_name}\" to answer this "
            "request. Do not answer in plain text."
        )
    elif isinstance(tool_choice, str) and tool_choice == "required":
        directive = (
            "You MUST call one of the available functions to answer this "
            "request. Do not answer in plain text."
        )
    else:
        directive = (
            "If (and only if) one of the available functions is appropriate for "
            "the user's request, call it. Otherwise, answer normally in plain "
            "text without emitting a tool-call block."
        )

    envelope_example = (
        "```json\n"
        '{\n'
        f'  "{ENVELOPE_KEY}": [\n'
        '    {"name": "<function name>", "arguments": {"<arg>": "<value>"}}\n'
        "  ]\n"
        "}\n"
        "```"
    )

    fragment = (
        "# Function calling\n"
        "You have access to the following functions:\n\n"
        f"{tool_descriptions}\n\n"
        f"{directive}\n\n"
        "To call a function, respond with ONLY a single fenced code block "
        "tagged as json containing an object with a "
        f'"{ENVELOPE_KEY}" array. Each element must have a "name" (the function '
        'name) and an "arguments" object whose keys/values follow that '
        "function's JSON schema. Emit no prose before or after the block when "
        "calling a function. Example:\n\n"
        f"{envelope_example}\n\n"
        "You may include more than one element in the array to request multiple "
        "function calls at once. The arguments object must be valid JSON."
    )

    return fragment


def _generate_call_id() -> str:
    """Generate an OpenAI-style tool-call id (``call_<uuid hex>``)."""
    return f"call_{uuid.uuid4().hex}"


def _extract_envelope_objects(text: str) -> List[Dict[str, Any]]:
    """
    Find and parse candidate JSON envelope objects in Claude's raw text.

    Looks first for fenced ```json blocks, then falls back to any fenced code
    block, and finally to a bare JSON object. Returns the list of parsed dicts
    that contain the envelope key.
    """
    candidates: List[str] = []

    # Prefer ```json ... ``` fenced blocks.
    fenced_json = re.findall(r"```json\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(fenced_json)

    # Fall back to any fenced ``` ... ``` block.
    if not candidates:
        fenced_any = re.findall(r"```\s*(.*?)```", text, flags=re.DOTALL)
        candidates.extend(fenced_any)

    # Fall back to the first bare {...} object spanning the text.
    if not candidates:
        brace_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if brace_match:
            candidates.append(brace_match.group(0))

    parsed: List[Dict[str, Any]] = []
    for raw in candidates:
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict) and ENVELOPE_KEY in obj:
            parsed.append(obj)

    return parsed


def parse_tool_calls(text: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """
    Parse Claude's raw text response for a tool-call envelope.

    Args:
        text: Claude's raw text response.

    Returns:
        A list of OpenAI ``tool_calls`` dicts when a valid envelope with at
        least one well-formed call is found, otherwise ``None`` (indicating the
        response should be treated as plain text).

        Each returned dict has the shape::

            {
                "id": "call_<hex>",
                "type": "function",
                "function": {"name": "...", "arguments": "<JSON string>"},
            }
    """
    if not text:
        return None

    envelopes = _extract_envelope_objects(text)
    if not envelopes:
        return None

    tool_calls: List[Dict[str, Any]] = []
    for envelope in envelopes:
        raw_calls = envelope.get(ENVELOPE_KEY)
        if not isinstance(raw_calls, list):
            continue
        for raw_call in raw_calls:
            if not isinstance(raw_call, dict):
                continue
            name = raw_call.get("name")
            if not name or not isinstance(name, str):
                continue

            arguments = raw_call.get("arguments", {})
            # OpenAI's ``arguments`` field is always a JSON *string*.
            if isinstance(arguments, str):
                arguments_str = arguments
            else:
                try:
                    arguments_str = json.dumps(arguments)
                except (TypeError, ValueError):
                    arguments_str = "{}"

            tool_calls.append(
                {
                    "id": _generate_call_id(),
                    "type": "function",
                    "function": {"name": name, "arguments": arguments_str},
                }
            )

    if not tool_calls:
        return None

    logger.info(f"Parsed {len(tool_calls)} prompt-based tool call(s) from Claude response")
    return tool_calls
