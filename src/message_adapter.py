from typing import List, Optional, Dict, Any
from src.models import Message
import re


class MessageAdapter:
    """Converts between OpenAI message format and Claude Code prompts."""

    @staticmethod
    def messages_to_prompt(messages: List[Message]) -> tuple[str, Optional[str]]:
        """
        Convert OpenAI messages to Claude Code prompt format.
        Returns (prompt, system_prompt)
        """
        system_parts = []
        conversation_parts = []

        for message in messages:
            if message.role == "system":
                # Preserve every system message. Clients commonly split the
                # system prompt and additional context (e.g. environment info,
                # dates, user identity) into separate system messages; keeping
                # only the last one silently strips that context.
                if message.content:
                    system_parts.append(message.content)
            elif message.role == "user":
                conversation_parts.append(f"Human: {message.content}")
            elif message.role == "assistant":
                conversation_parts.append(f"Assistant: {message.content}")

        # Combine all system messages, preserving order.
        system_prompt = "\n\n".join(system_parts) if system_parts else None

        # Join conversation parts
        prompt = "\n\n".join(conversation_parts)

        # If the last message wasn't from the user, add a prompt for assistant
        if messages and messages[-1].role != "user":
            prompt += "\n\nHuman: Please continue."

        return prompt, system_prompt

    @staticmethod
    def filter_content(content: str) -> str:
        """
        Filter content for unsupported features and tool usage.
        Remove thinking blocks, tool calls, and image references.
        """
        if not content:
            return content

        # Remove thinking blocks (common when tools are disabled but Claude tries to think)
        thinking_pattern = r"<thinking>.*?</thinking>"
        content = re.sub(thinking_pattern, "", content, flags=re.DOTALL)

        # Extract content from attempt_completion blocks (these contain the actual user response)
        attempt_completion_pattern = r"<attempt_completion>(.*?)</attempt_completion>"
        attempt_matches = re.findall(attempt_completion_pattern, content, flags=re.DOTALL)
        if attempt_matches:
            # Use the content from the attempt_completion block
            extracted_content = attempt_matches[0].strip()

            # If there's a <result> tag inside, extract from that
            result_pattern = r"<result>(.*?)</result>"
            result_matches = re.findall(result_pattern, extracted_content, flags=re.DOTALL)
            if result_matches:
                extracted_content = result_matches[0].strip()

            if extracted_content:
                content = extracted_content
        else:
            # Remove other tool usage blocks (when tools are disabled but Claude tries to use them)
            tool_patterns = [
                r"<read_file>.*?</read_file>",
                r"<write_file>.*?</write_file>",
                r"<bash>.*?</bash>",
                r"<search_files>.*?</search_files>",
                r"<str_replace_editor>.*?</str_replace_editor>",
                r"<args>.*?</args>",
                r"<ask_followup_question>.*?</ask_followup_question>",
                r"<attempt_completion>.*?</attempt_completion>",
                r"<question>.*?</question>",
                r"<follow_up>.*?</follow_up>",
                r"<suggest>.*?</suggest>",
            ]

            for pattern in tool_patterns:
                content = re.sub(pattern, "", content, flags=re.DOTALL)

        # Pattern to match image references or base64 data
        image_pattern = r"\[Image:.*?\]|data:image/.*?;base64,.*?(?=\s|$)"

        def replace_image(match):
            return "[Image: Content not supported by Claude Code]"

        content = re.sub(image_pattern, replace_image, content)

        # Clean up extra whitespace and newlines
        content = re.sub(r"\n\s*\n\s*\n", "\n\n", content)  # Multiple newlines to double
        content = content.strip()

        # If content is now empty or only whitespace, provide a fallback
        if not content or content.isspace():
            return "I understand you're testing the system. How can I help you today?"

        return content

    @staticmethod
    def filter_content_streaming(content: str) -> str:
        """Filter a streamed text block for unsupported features.

        This is a streaming-safe sibling of :meth:`filter_content`. It applies
        the same thinking/tool/image scrubbing, but:

        * It never substitutes the "I understand you're testing the system..."
          fallback. During streaming we must be able to emit an empty string for
          a chunk that scrubs down to nothing (the caller is responsible for
          skipping empty deltas), otherwise placeholder text would leak into the
          middle of a real response.
        * It does not ``strip()`` the result, so inter-token whitespace and
          newlines between incremental deltas are preserved on the wire.

        Returns the filtered text, which may be an empty string.
        """
        if not content:
            return ""

        # Remove thinking blocks (Claude may emit these even when tools are off).
        content = re.sub(r"<thinking>.*?</thinking>", "", content, flags=re.DOTALL)

        # Prefer the user-facing content inside an attempt_completion block.
        attempt_matches = re.findall(
            r"<attempt_completion>(.*?)</attempt_completion>", content, flags=re.DOTALL
        )
        if attempt_matches:
            extracted_content = attempt_matches[0].strip()
            result_matches = re.findall(
                r"<result>(.*?)</result>", extracted_content, flags=re.DOTALL
            )
            if result_matches:
                extracted_content = result_matches[0].strip()
            content = extracted_content
        else:
            tool_patterns = [
                r"<read_file>.*?</read_file>",
                r"<write_file>.*?</write_file>",
                r"<bash>.*?</bash>",
                r"<search_files>.*?</search_files>",
                r"<str_replace_editor>.*?</str_replace_editor>",
                r"<args>.*?</args>",
                r"<ask_followup_question>.*?</ask_followup_question>",
                r"<attempt_completion>.*?</attempt_completion>",
                r"<question>.*?</question>",
                r"<follow_up>.*?</follow_up>",
                r"<suggest>.*?</suggest>",
            ]
            for pattern in tool_patterns:
                content = re.sub(pattern, "", content, flags=re.DOTALL)

        image_pattern = r"\[Image:.*?\]|data:image/.*?;base64,.*?(?=\s|$)"
        content = re.sub(image_pattern, "[Image: Content not supported by Claude Code]", content)

        return content

    @staticmethod
    def segment_text(text: str, max_chunk_size: int = 0) -> List[str]:
        """Split a single text block into reasonably sized streaming deltas.

        Used to smooth a very large assistant block into multiple SSE deltas so
        clients receive incremental output instead of one giant payload. When
        ``max_chunk_size`` is ``0`` (the default) the text is returned unchanged
        as a single-element list.

        Splitting is whitespace-aware: it breaks on the last whitespace boundary
        within the window when possible to avoid cutting words mid-token, and
        never emits empty segments.
        """
        if max_chunk_size <= 0 or len(text) <= max_chunk_size:
            return [text] if text else []

        segments: List[str] = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + max_chunk_size, length)
            if end < length:
                # Try to break on a whitespace boundary inside the window.
                boundary = text.rfind(" ", start, end)
                if boundary <= start:
                    boundary = text.rfind("\n", start, end)
                if boundary > start:
                    end = boundary + 1
            segment = text[start:end]
            if segment:
                segments.append(segment)
            start = end
        return segments

    @staticmethod
    def format_claude_response(
        content: str, model: str, finish_reason: str = "stop"
    ) -> Dict[str, Any]:
        """Format Claude response for OpenAI compatibility."""
        return {
            "role": "assistant",
            "content": content,
            "finish_reason": finish_reason,
            "model": model,
        }

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Rough estimation of token count.
        OpenAI's rule of thumb: ~4 characters per token for English text.
        """
        return len(text) // 4
