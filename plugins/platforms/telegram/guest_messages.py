"""Telegram Bot API 10.0 guest messages: the bot summoned by an @mention in a chat it is not a member of.

Telegram delivers the mention as ``Update.guest_message`` with a ``guest_query_id``. The bot may answer it
exactly once (``answerGuestQuery``) and afterwards only edit the resulting inline message; it cannot
sendMessage into that chat. So a guest turn is one inline message.

The turn rides a *guest lane*: the query is answered at once with a placeholder and the turn's ``thread_id``
becomes ``guest_<inline_message_id>``. The gateway already carries ``thread_id`` in the metadata of every
send/edit/typing call of a turn (stream edits, progress, status, final reply), so the overrides below
recognise the lane and edit that single message. ``thread_id`` is also part of the session key, so every
guest turn gets its own session: no transcript leaks between callers or invocations, and no gateway-core
change is needed.

Kept as a subclass that ``_build_adapter`` constructs, so the feature touches one line of ``adapter.py``.
Unrelated to ``telegram.guest_mode`` (the @mention bypass for non-allowlisted groups the bot IS in).
Placeholder text: ``telegram.extra.guest_thinking_text``.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional

from gateway.platforms.base import SendResult, _prefix_within_utf16_limit, classify_send_error, utf16_len
from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
from plugins.platforms.telegram.adapter import TelegramAdapter

logger = logging.getLogger(__name__)

GUEST_LANE_PREFIX = "guest_"
_DEFAULT_THINKING_TEXT = "\U0001f4ad Thinking..."
_TRUNCATED_NOTE = "\n\n[Response truncated: a Telegram guest reply is a single message.]"


def guest_inline_message_id(metadata: Optional[Dict[str, Any]] = None, message_id: Any = None) -> Optional[str]:
    """The inline message id of the guest lane a call targets, else None. ``message_id`` covers the core's
    metadata-less edits of a turn's own message (final stream edit, heartbeat, queued-lane reconcile)."""
    for value in ((metadata or {}).get("thread_id"), message_id):
        if isinstance(value, str) and value.startswith(GUEST_LANE_PREFIX):
            return value[len(GUEST_LANE_PREFIX):]
    return None


def _is_guest_event(event: MessageEvent) -> bool:
    return guest_inline_message_id({"thread_id": getattr(event.source, "thread_id", None)}) is not None


def _clamp_to_one_message(content: str, limit: int) -> str:
    if utf16_len(content) <= limit:
        return content
    return _prefix_within_utf16_limit(content, limit - utf16_len(_TRUNCATED_NOTE)).rstrip() + _TRUNCATED_NOTE


class GuestTelegramAdapter(TelegramAdapter):
    """``TelegramAdapter`` that also serves guest messages.

    Every chat-scoped Bot API call stays off a guest lane: the bot is not in that chat, and PTB documents
    that a guest chat's id may coincide with another chat of this bot (e.g. its DM with that user), so a
    stray typing action, draft or reaction could land there.
    """

    def _register_handlers(self, app) -> None:
        from telegram.ext import MessageHandler, filters

        # First in group 0: PTB's message filters also match update.guest_message and the first match in a
        # group wins, so the member-chat handlers registered next never see a guest message.
        app.add_handler(MessageHandler(filters.UpdateType.GUEST_MESSAGE, self._handle_guest_message))
        super()._register_handlers(app)

    async def _handle_guest_message(self, update: Any, context: Any) -> None:
        msg = update.guest_message
        if not msg or not msg.guest_query_id or not (msg.text or msg.caption or "").strip():
            return
        user = msg.from_user
        # Fail closed BEFORE answering, with the runner's own auth chain: a query left unanswered shows the
        # caller nothing, but an answered one whose turn the runner then refuses keeps the placeholder forever.
        if user is None or not self._is_callback_user_authorized(
                str(user.id), chat_id=str(msg.chat.id), chat_type=self._chat_type_str(msg.chat),
                user_name=user.username or user.full_name):
            self._log_blocked_user(msg, what="guest caller")
            return
        from telegram import InlineQueryResultArticle, InputTextMessageContent

        placeholder = str(self.config.extra.get("guest_thinking_text") or _DEFAULT_THINKING_TEXT)
        try:
            sent = await self._bot.answer_guest_query(msg.guest_query_id, InlineQueryResultArticle(
                id=uuid.uuid4().hex, title="Hermes", input_message_content=InputTextMessageContent(placeholder)))
        except Exception as exc:
            logger.warning("[%s] answerGuestQuery failed for chat %s: %s", self.name, msg.chat.id, exc)
            return
        event = await self._build_triggered_event(msg, update, MessageType.TEXT)
        event.source.thread_id = GUEST_LANE_PREFIX + sent.inline_message_id
        await self.handle_message(event)

    async def _edit_guest_message(self, inline_message_id: str, content: str) -> SendResult:
        """Show ``content`` in the lane's inline message: MarkdownV2 first, plain text if Telegram rejects it."""
        from telegram.constants import ParseMode
        from telegram.error import BadRequest

        lane_id = GUEST_LANE_PREFIX + inline_message_id
        plain = _clamp_to_one_message(content, self.MAX_MESSAGE_LENGTH)
        formatted = self.format_message(plain)
        attempts = [(plain, None)]
        if utf16_len(formatted) <= self.MAX_MESSAGE_LENGTH:
            attempts.insert(0, (formatted, ParseMode.MARKDOWN_V2))
        error: Optional[Exception] = None
        for text, parse_mode in attempts:
            try:
                await self._bot.edit_message_text(
                    inline_message_id=inline_message_id, text=text, parse_mode=parse_mode,
                    **self._link_preview_kwargs())
            except BadRequest as exc:
                if "not modified" not in str(exc).lower():
                    error = exc
                    continue
            except Exception as exc:
                error = exc
                break
            return SendResult(success=True, message_id=lane_id)
        kind = classify_send_error(error)
        logger.warning("[%s] Failed to edit Telegram guest message %s: %s", self.name, inline_message_id, error)
        return SendResult(
            success=False, error=str(error), error_kind=kind, retryable=kind in {"transient", "rate_limited"})

    async def send(self, chat_id, content, reply_to=None, metadata=None, **kwargs) -> SendResult:
        guest_id = guest_inline_message_id(metadata)
        if guest_id is None:
            return await super().send(chat_id, content, reply_to, metadata, **kwargs)
        if not content or not content.strip():
            return SendResult(success=True, message_id=None)
        return await self._edit_guest_message(guest_id, content)

    async def edit_message(
            self, chat_id, message_id, content, *, finalize=False, metadata=None, **kwargs) -> SendResult:
        guest_id = guest_inline_message_id(metadata, message_id)
        if guest_id is None:
            return await super().edit_message(
                chat_id, message_id, content, finalize=finalize, metadata=metadata, **kwargs)
        return await self._edit_guest_message(guest_id, content)

    async def send_or_update_status(self, chat_id, status_key, content, *, metadata=None, **kwargs) -> SendResult:
        # Never cache a guest id under (chat_id, status_key): a coinciding chat of this bot would edit it.
        guest_id = guest_inline_message_id(metadata)
        if guest_id is None:
            return await super().send_or_update_status(chat_id, status_key, content, metadata=metadata, **kwargs)
        return await self._edit_guest_message(guest_id, content)

    async def send_typing(self, chat_id, metadata=None) -> None:
        if guest_inline_message_id(metadata) is None:
            await super().send_typing(chat_id, metadata=metadata)

    def supports_draft_streaming(self, chat_type=None, metadata=None, **kwargs) -> bool:
        return guest_inline_message_id(metadata) is None and super().supports_draft_streaming(
            chat_type, metadata, **kwargs)

    async def on_processing_start(self, event: MessageEvent) -> None:
        if not _is_guest_event(event):
            await super().on_processing_start(event)

    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        if not _is_guest_event(event):
            await super().on_processing_complete(event, outcome)
