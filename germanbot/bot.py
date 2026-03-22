import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes
import sys
print(sys.executable)
load_dotenv()

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("flowise-german-bot")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

FLOWISE_BASE_URL = os.getenv("FLOWISE_BASE_URL", "").rstrip("/")
FLOWISE_CHATFLOW_ID = os.getenv("FLOWISE_CHATFLOW_ID", "")
FLOWISE_API_KEY = os.getenv("FLOWISE_API_KEY", "")

BOT_TIMEZONE = os.getenv("BOT_TIMEZONE", "Europe/Berlin")
DAILY_HOUR = int(os.getenv("DAILY_HOUR", "12"))
DAILY_MINUTE = int(os.getenv("DAILY_MINUTE", "0"))

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
if not TELEGRAM_CHAT_ID:
    raise RuntimeError("TELEGRAM_CHAT_ID is not set")
if not FLOWISE_BASE_URL:
    raise RuntimeError("FLOWISE_BASE_URL is not set")
if not FLOWISE_CHATFLOW_ID:
    raise RuntimeError("FLOWISE_CHATFLOW_ID is not set")


def build_question() -> str:
    return (
        "Get exactly one recent German news item from the configured website tool, "
        "rewrite it to CEFR A2 German, then retrieve exactly 20 relevant German words "
        "only from the vector database retriever tool. "
        "Do not extract vocabulary from the website text. "
        "Do not invent vocabulary. "
        "Return valid JSON only."
    )


async def call_flowise() -> Dict[str, Any]:
    """
    Calls Flowise Prediction API.
    """
    url = f"{FLOWISE_BASE_URL}/api/v1/prediction/{FLOWISE_CHATFLOW_ID}"

    headers = {"Content-Type": "application/json"}
    if FLOWISE_API_KEY:
        headers["Authorization"] = f"Bearer {FLOWISE_API_KEY}"

    payload = {
        "question": build_question(),
        "streaming": False,
        "overrideConfig": {
            "sessionId": "telegram-daily-german"
        }
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

    return parse_flowise_response(result)


def parse_flowise_response(result: Any) -> Dict[str, Any]:
    """
    Flowise can return JSON in different shapes.
    We try:
    1) result["json"]
    2) JSON inside result["text"]
    3) direct object
    """
    if isinstance(result, dict):
        if isinstance(result.get("json"), dict):
            return result["json"]

        if isinstance(result.get("text"), str):
            extracted = try_extract_json(result["text"])
            if extracted:
                return extracted

        # Sometimes result itself is already the final JSON
        if all(k in result for k in ["title", "url", "source_summary", "a2_text", "topic", "new_words"]):
            return result

    raise ValueError(f"Could not parse Flowise response: {result}")


def try_extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()

    # Direct JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # JSON inside ```json ... ```
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # Any first {...}
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return None


def escape_html(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def validate_words(words: Any) -> List[Dict[str, str]]:
    if not isinstance(words, list):
        return []

    valid_words: List[Dict[str, str]] = []
    for item in words:
        if not isinstance(item, dict):
            continue
        word = str(item.get("word", "")).strip()
        translation = str(item.get("translation", "")).strip()
        if word and translation:
            valid_words.append({
                "word": word,
                "translation": translation
            })
    return valid_words[:20]


def build_message(data: Dict[str, Any]) -> str:
    error = str(data.get("error", "")).strip()
    if error:
        return f"⚠️ Ошибка агента:\n<pre>{escape_html(error)}</pre>"

    title = escape_html(data.get("title", "Без заголовка"))
    url = str(data.get("url", "")).strip()
    source_summary = escape_html(data.get("source_summary", ""))
    a2_text = escape_html(data.get("a2_text", ""))
    topic = escape_html(data.get("topic", ""))
    words = validate_words(data.get("new_words", []))

    lines: List[str] = [f"🇩🇪 <b>{title}</b>"]

    if url:
        lines.append(f'<a href="{url}">Источник</a>')

    if topic:
        lines.append(f"\n<b>Тема:</b> {topic}")

    if source_summary:
        lines.append(f"\n<b>Кратко по новости:</b>\n{source_summary}")

    if a2_text:
        lines.append(f"\n<b>Версия A2:</b>\n{a2_text}")

    if words:
        lines.append("\n<b>20 слов из векторной базы:</b>")
        for i, item in enumerate(words, start=1):
            word = escape_html(item["word"])
            translation = escape_html(item["translation"])
            lines.append(f"{i}. <b>{word}</b> — {translation}")
    else:
        lines.append("\n<b>20 слов из векторной базы:</b>\n— не получены")

    return "\n".join(lines)


def split_message(text: str, max_len: int = 4000) -> List[str]:
    if len(text) <= max_len:
        return [text]

    parts: List[str] = []
    current = ""

    for line in text.split("\n"):
        if len(current) + len(line) + 1 > max_len:
            if current:
                parts.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line

    if current:
        parts.append(current)

    return parts


async def send_daily_news(application: Application) -> None:
    try:
        data = await call_flowise()
        message = build_message(data)
    except Exception as e:
        logger.exception("Failed to send daily news")
        message = f"❌ Ошибка при получении новости:\n<pre>{escape_html(str(e))}</pre>"

    for part in split_message(message):
        await application.bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=part,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=False,
        )


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Бот запущен.\n"
        "/now — получить новость сейчас\n"
        "/chatid — показать chat id"
    )


async def now_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Запрашиваю новость у Flowise...")
    try:
        data = await call_flowise()
        message = build_message(data)
        for part in split_message(message):
            await update.message.reply_text(
                part,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=False,
            )
    except Exception as e:
        logger.exception("Failed on /now")
        await update.message.reply_text(f"Ошибка: {e}")


async def chatid_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat:
        await update.message.reply_text(str(update.effective_chat.id))


async def post_init(application: Application) -> None:
    scheduler = AsyncIOScheduler(timezone=BOT_TIMEZONE)
    scheduler.add_job(
        send_daily_news,
        trigger="cron",
        hour=DAILY_HOUR,
        minute=DAILY_MINUTE,
        kwargs={"application": application},
        id="daily_news_job",
        replace_existing=True,
    )
    scheduler.start()
    logger.info(
        "Scheduler started for %02d:%02d timezone=%s",
        DAILY_HOUR,
        DAILY_MINUTE,
        BOT_TIMEZONE,
    )


def main() -> None:
    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("now", now_command))
    app.add_handler(CommandHandler("chatid", chatid_command))

    logger.info("Bot started")
    app.run_polling()


if __name__ == "__main__":
    main()