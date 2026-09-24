"""Bot API 10.0 guest messages: real PTB dispatch, the runner-built send metadata, and the real session key."""

import dataclasses
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("telegram")
from telegram import Chat, Message, PhotoSize, Update, User
from telegram.ext import Application

from gateway.config import PlatformConfig
from gateway.platforms.base import _thread_metadata_for_source, utf16_len
from gateway.run import GatewayRunner
from gateway.session import build_session_key
from plugins.platforms.telegram.adapter import _build_adapter

CHAT = Chat(-1001234, "supergroup", title="Somebody else's group")
CALLER = User(42, "Alice", False, username="alice")


def _guest_update(update_id: int, *, text=None, caption=None, photo=None) -> Update:
    msg = Message(
        update_id, datetime.now(timezone.utc), CHAT, from_user=CALLER, text=text, caption=caption,
        photo=photo, guest_query_id=f"gq{update_id}")
    return Update(update_id, guest_message=msg)


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "42")
    adapter = _build_adapter(PlatformConfig(enabled=True, token="123:ABC"))
    adapter._bot = AsyncMock(username="hermes_bot")
    adapter._bot.answer_guest_query.side_effect = [
        SimpleNamespace(inline_message_id="IM1"), SimpleNamespace(inline_message_id="IM2")]
    adapter.handle_message = AsyncMock()
    return adapter


def test_guest_updates_never_reach_member_chat_handlers(adapter):
    app = Application.builder().token("123:ABC").build()
    adapter._register_handlers(app)
    for update in (_guest_update(1, text="@hermes_bot hi"),
                   _guest_update(2, caption="@hermes_bot what is this?", photo=[PhotoSize("f", "u", 1, 1)])):
        claimed = next(h for h in app.handlers[0] if h.check_update(update))
        assert claimed.callback == adapter._handle_guest_message


@pytest.mark.asyncio
async def test_unauthorized_guest_caller_gets_no_answer_and_no_turn(adapter, monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "7")
    await adapter._handle_guest_message(_guest_update(1, text="@hermes_bot hi"), None)
    adapter._bot.answer_guest_query.assert_not_awaited()
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_each_guest_turn_is_its_own_session_and_every_send_edits_its_inline_message(adapter):
    for update_id in (1, 2):
        await adapter._handle_guest_message(_guest_update(update_id, text="@hermes_bot hi"), None)
    first, second = (call.args[0].source for call in adapter.handle_message.await_args_list)
    member_turn = dataclasses.replace(first, thread_id=None)
    assert len({build_session_key(first), build_session_key(second), build_session_key(member_turn)}) == 3

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {}
    for source, inline_id in ((first, "IM1"), (second, "IM2")):
        for metadata in (_thread_metadata_for_source(source), runner._thread_metadata_for_source(source)):
            adapter._bot.edit_message_text.reset_mock()
            sent = await adapter.send(source.chat_id, "partial", reply_to=source.message_id, metadata=metadata)
            await adapter.edit_message(source.chat_id, sent.message_id, "final", finalize=True, metadata=metadata)
            await adapter.edit_message(source.chat_id, sent.message_id, "final, no metadata")
            await adapter.send_typing(source.chat_id, metadata=metadata)
            targets = [call.kwargs["inline_message_id"] for call in adapter._bot.edit_message_text.await_args_list]
            assert sent.success and targets == [inline_id] * 3
            assert not adapter.supports_draft_streaming("dm", metadata)
    # The guest chat's id may coincide with another chat of this bot: nothing chat-scoped may reach it.
    adapter._bot.send_message.assert_not_awaited()
    adapter._bot.send_chat_action.assert_not_awaited()


@pytest.mark.asyncio
async def test_guest_reply_is_clamped_to_one_message(adapter):
    result = await adapter.send("-1001234", "\U0001f600" * 3000, metadata={"thread_id": "guest_IM1"})
    (call,) = adapter._bot.edit_message_text.await_args_list
    assert result.success and utf16_len(call.kwargs["text"]) <= adapter.MAX_MESSAGE_LENGTH


@pytest.mark.asyncio
async def test_a_coinciding_member_chat_never_edits_a_guest_message(adapter):
    adapter._bot.send_message.return_value = SimpleNamespace(message_id=77)
    await adapter.send_or_update_status("42", "context", "guest status", metadata={"thread_id": "guest_IM1"})
    adapter._bot.edit_message_text.reset_mock()
    await adapter.send_or_update_status("42", "context", "DM status")
    assert all("inline_message_id" not in call.kwargs for call in adapter._bot.edit_message_text.await_args_list)
    adapter._bot.send_message.assert_awaited()
