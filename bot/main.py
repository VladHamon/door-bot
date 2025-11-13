from google import genai
from google.genai import types
import os, json, io, uuid, re, asyncio, textwrap
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import CommandStart
from aiogram.enums import ChatAction
from aiogram.types import (
    Message,
    CallbackQuery,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    BufferedInputFile,
    FSInputFile,
    InputMediaPhoto,
)
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext
from PIL import Image
import httpx

# =========================== ENV ===========================
load_dotenv()
BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "")
REQUIRED_BUILDER2112 = os.getenv("REQUIRED_CHANNEL", "@yourdoorshop")

bot = Bot(BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())
router = Router()
dp.include_router(router)

CATALOG = json.loads(Path("catalog.json").read_text(encoding="utf-8"))

# =========================== FSM ===========================
class Flow(StatesGroup):
    waiting_foto = State()
    describing = State()
    selecting_door = State()
    selecting_color = State()
    generating = State()
    after_result = State()

# =========================== UTILS ===========================
async def ensure_subscribed(user_id: int) -> bool:
    try:
        member = await bot.get_chat_member(REQUIRED_BUILDER2112, user_id)
        return getattr(member, "status", None) in ("member", "creator", "administrator")
    except Exception:
        return False

async def tg_download_photo(message: Message, dest: Path) -> Path:
    photo = max(message.photo, key=lambda p: getattr(p, "file_size", 0))
    f = await bot.get_file(photo.file_id)
    url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{f.file_path}"
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.content
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dest

def parse_color(s: str) -> str:
    """
    Принимает: #HEX, 'RAL 9010', 'white', 'beige'...
    Возвращает исходную строку (нормализованную) или HEX-мэппинг для базовых слов.
    """
    if not s:
        return ""
    s = s.strip()
    if re.match(r"^#([0-9A-Fa-f]{6})$", s):
        return s.upper()
    if re.match(r"^(RAL\s*\d{3,4})$", s, re.IGNORECASE):
        # Нормализация: 'ral 9010' -> 'RAL 9010'
        dig = re.findall(r"\d{3,4}", s)[0]
        return f"RAL {dig}"
    basic = {
        "white": "#FFFFFF", "black": "#000000", "beige": "#E6D8C3", "cream": "#F3F0E6",
        "gray": "#BFBFBF", "light gray": "#D9D9D9", "dark gray": "#6B6B6B",
        "oak": "#D8C4A6", "walnut": "#8B6A4E", "green": "#2F5A3C", "brown": "#6B4E2E"
    }
    return basic.get(s.lower(), s)

def truncate(s: str, limit: int = 3500) -> str:
    """Безопасная обрезка длинных текстов для телеграм-капшенов/сообщений."""
    s = s.strip()
    return s if len(s) <= limit else s[:limit-3] + "..."

# =========================== CHAT ACTIONS ===========================
async def run_chat_action(chat_id: int, action: ChatAction, stop_event: asyncio.Event, interval: float = 4.0):
    """
    Периодически шлём статус 'typing' / 'upload_photo' пока идёт долгая операция,
    чтобы пользователь видел интерактив.
    """
    try:
        while not stop_event.is_set():
            await bot.send_chat_action(chat_id, action)
            await asyncio.sleep(interval)
    except Exception:
        # Безопасно глотаем исключения — индикатор только UI-украшение
        pass

# =========================== PARSERS ===========================
def extract_json_block(text: str) -> Optional[dict]:
    """
    Ищем последний JSON-блок в ответе (на случай, если модель добавила лишний текст).
    """
    if not text:
        return None
    # Уберём возможные тройные кавычки/маркдауны
    clean = text.replace("```json", "```").replace("```", "")
    # Пытаемся найти последний блок {...}
    last_open = clean.rfind("{")
    last_close = clean.rfind("}")
    if last_open == -1 or last_close == -1 or last_close < last_open:
        return None
    candidate = clean[last_open:last_close+1]
    try:
        return json.loads(candidate)
    except Exception:
        # fallback — ищем все объекты {}
        objs = re.findall(r"\{[\s\S]*?\}", clean)
        for raw in reversed(objs):
            try:
                return json.loads(raw)
            except Exception:
                continue
    return None

def normalize_recommended_colors(j: Optional[dict]) -> List[Dict[str, str]]:
    """
    Поддерживаем несколько ключей на случай вариативности модели.
    Ожидаем формат:
    {"recommended_door_colors": [{"name": "...", "ral": "RAL 9016", "reason_ru": "...", "hex": "#FFFFFF"}]}
    """
    if not j:
        return []
    keys = ["recommended_door_colors", "recommended_colors", "door_colors", "colors"]
    arr: List[dict] = []
    for k in keys:
        if isinstance(j.get(k), list):
            arr = j[k]
            break
    result: List[Dict[str, str]] = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("color_name") or ""
        ral = item.get("ral") or item.get("RAL") or ""
        reason_ru = item.get("reason_ru") or item.get("reason") or ""
        hexv = item.get("hex") or item.get("HEX") or ""
        out = {}
        if name: out["name"] = str(name)
        if ral: out["ral"] = parse_color(str(ral))
        if reason_ru: out["reason_ru"] = str(reason_ru)
        if hexv and re.match(r"^#([0-9A-Fa-f]{6})$", hexv): out["hex"] = hexv.upper()
        result.append(out)
    # Уникализируем по (ral or name or hex)
    seen = set()
    uniq = []
    for c in result:
        key = c.get("ral") or c.get("name") or c.get("hex") or json.dumps(c, sort_keys=True)
        if key not in seen:
            uniq.append(c)
            seen.add(key)
    return uniq[:8]  # ограничим до разумного количества

# =========================== GEMINI HELPERS ===========================
def _resp_text(resp) -> str:
    # Универсальное извлечение текста из ответа Gemini
    if getattr(resp, "text", None):
        return resp.text
    if hasattr(resp, "candidates") and resp.candidates:
        parts = []
        for c in resp.candidates:
            if getattr(c, "content", None) and getattr(c.content, "parts", None):
                for p in c.content.parts:
                    t = getattr(p, "text", None)
                    if t:
                        parts.append(t)
        if parts:
            return "\n".join(parts)
    return ""

def _resp_image_bytes(resp) -> bytes:
    # Универсальное извлечение байтов изображения из ответа Gemini
    if hasattr(resp, "parts"):
        for part in resp.parts:
            if getattr(part, "inline_data", None) and getattr(part.inline_data, "data", None):
                return part.inline_data.data
            if hasattr(part, "as_image"):
                try:
                    pil = part.as_image()
                    buf = io.BytesIO()
                    pil.save(buf, format="PNG")
                    return buf.getvalue()
                except Exception:
                    pass
    if hasattr(resp, "candidates") and resp.candidates:
        cand = resp.candidates[0]
        if hasattr(cand, "content") and getattr(cand.content, "parts", None):
            for p in cand.content.parts:
                if getattr(p, "inline_data", None) and getattr(p.inline_data, "data", None):
                    return p.inline_data.data
    # Попытка через dict
    if hasattr(resp, "to_dict"):
        d = resp.to_dict()
        data = (
            d.get("candidates", [{}])[0]
             .get("content", {})
             .get("parts", [{}])[0]
             .get("inline_data", {})
             .get("data")
        )
        if data:
            return data
    raise RuntimeError("Gemini did not return an image payload")

# =========================== 1) ОПИСАНИЕ СЦЕНЫ (Gemini 2.5 Pro) ===========================
async def describe_scene_with_gemini(image_path: Path) -> Tuple[str, List[Dict[str, str]]]:
    """
    Возвращает:
      - english_description: STR (максимально подробное описание интерьера без дверей/проёмов/локаций/формы комнаты)
      - recommended_colors: List[{"name","ral","hex?","reason_ru"}]
    """
    client = genai.Client(api_key=GEMINI_API_KEY)

    schema_prompt = textwrap.dedent("""
        Describe this interior as thoroughly as possible. Style and type. Capture absolutely everything — every single detail —
        including all colors and the full color palette (Accuracy in the rendering of color and materials is very 
        important; the color must be described in such a way that any artist can easily draw identical materials based 
        on the description), interior objects with their shapes, sizes, and types,
        the lighting, the floor (type, texture, material, and color), the walls (material and color),
        the ceiling, and so on down to the smallest element. If the scene contains tiles, parquet, patterns on the wall, patterns on the floor, 
        their exact size must be indicated.
        In the description, you MUST NOT mention doors, doorways, or anything related to them.
        The description MUST NOT include the location of interior items, the shape of the room,
        or the location of anything in the interior at all.
        Write the description in English.
    """).strip()

    img = Image.open(image_path).convert("RGB")

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[schema_prompt, img],
        config=types.GenerateContentConfig(temperature=0.2),
    )

    txt = _resp_text(resp).strip()

    # Парсим JSON-блок с рекомендациями
    j = extract_json_block(txt)
    recommended = normalize_recommended_colors(j)

    # Убираем JSON из текста, оставляя англ. описание
    english_description = txt
    if j:
        # попытка "вырезать" JSON из конца
        try:
            dumped = json.dumps(j, ensure_ascii=False)
            cut_pos = english_description.rfind(dumped)
            if cut_pos != -1:
                english_description = english_description[:cut_pos].strip()
        except Exception:
            pass

    return english_description, recommended

# =========================== 2) ГЕНЕРАЦИЯ КАДРА (Gemini 2.5 Flash Image) ===========================
def build_generation_prompt(interior_en: str, door_color_text: str) -> str:
    """
    Собираем единый промпт (строго по ТЗ).
    door_color_text — строка, которую выбрал пользователь (например, "RAL 9016 Traffic White" или "#E6D8C3 beige").
    """
    interior_block = interior_en.strip()
    door_color_line = door_color_text.strip() or "a neutral light tone"

    return f"""
Create an ULTRA-REALISTIC interior photograph by RECONSTRUCTING the room from the following text ONLY (no base photo is provided).
Then INSERT exactly ONE door leaf using the attached DOOR IMAGE.

CRITICAL CONSTRAINTS (must be followed precisely):
- The inserted DOOR is the SINGLE, PRIMARY, and CENTRAL visual subject of the image.
- The DOOR must be placed on the BACK WALL, centered in the composition (one-point perspective).
- The DOOR must be seen FULLY and DIRECTLY from the front (no partial view, no angle cuts).
- NOTHING may be in front of, across, or partially overlapping the door — not even slightly.
- The door must be COMPLETELY VISIBLE from top to bottom and from edge to edge of the frame.
- If any object (furniture, plant, decor, curtain, light fixture, etc.) partially blocks or touches the door,
  the generation is considered INCORRECT.
- The area in front of the door must remain EMPTY and CLEAR, ensuring 100% unobstructed visibility.

DOOR (hard constraints):
- Use the attached DOOR IMAGE as the ONLY door. Keep its exact geometry (panel layout), proportions, and hardware.
- Recolor the DOOR LEAF and DOOR FRAMES (panel surfaces only) to: {door_color_line}. Do NOT recolor metal hardware.
- The door occupies the exact center of the image, on the back wall, viewed frontally.
- No other doors, arches, or openings exist anywhere in the scene.

ROOM:
{interior_block}

QUALITY:
- Photorealistic PBR shading; correct perspective; clean global illumination; accurate color management; minimal noise.
- Balanced exposure, no HDR halos, no over-sharpening, no text or people.

""".strip()

async def gemini_generate(door_png: Path, color_text: str, interior_en: str, aspect: str = "3:4") -> bytes:
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = build_generation_prompt(interior_en=interior_en, door_color_text=color_text)
    img = Image.open(door_png).convert("RGBA")

    cfg = types.GenerateContentConfig(
        response_modalities=["Image"],
        image_config=types.ImageConfig(aspect_ratio=aspect),
        temperature=0.4,
        top_p=0.5,
    )

    resp = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[prompt, img],
        config=cfg,
    )
    return _resp_image_bytes(resp)

# =========================== UI BUILDERS ===========================
def build_colors_keyboard_and_text(colors: List[Dict[str, str]]) -> Tuple[InlineKeyboardMarkup, str]:
    """
    Строим кнопки по цветам + краткий поясняющий текст (на русском) над кнопками.
    """
    rows = []
    description_lines = []
    # Покажем не более 6 для компактности
    for idx, c in enumerate(colors[:6]):
        name = c.get("name", "").strip()
        ral = c.get("ral", "").strip()
        hexv = c.get("hex", "").strip()
        reason = c.get("reason_ru", "").strip()
        label_parts = []
        if name: label_parts.append(name)
        if ral: label_parts.append(ral)
        if not label_parts and hexv:
            label_parts.append(hexv)
        label = " / ".join(label_parts) if label_parts else f"Color {idx+1}"
        rows.append([InlineKeyboardButton(text=label, callback_data=f"color_idx:{idx}")])
        if reason:
            description_lines.append(f"• {label}: {reason}")
    # Добавим кнопку «Другой цвет»
    rows.append([InlineKeyboardButton(text="🎨 Ввести свой цвет…", callback_data="color:custom")])
    kb = InlineKeyboardMarkup(inline_keyboard=rows)
    description_text = "\n".join(description_lines) if description_lines else "Выберите один из предложенных оттенков или введите свой цвет."
    return kb, description_text

def current_catalog_index(state_data: Dict[str, Any]) -> int:
    return int(state_data.get("carousel_idx", 0))

def build_carousel_keyboard(idx: int) -> InlineKeyboardMarkup:
    nav = [
        InlineKeyboardButton(text="◀", callback_data="carousel:prev"),
        InlineKeyboardButton(text="✅ Выбрать", callback_data="carousel:choose"),
        InlineKeyboardButton(text="▶", callback_data="carousel:next"),
    ]
    return InlineKeyboardMarkup(inline_keyboard=[nav])

def door_caption(door: Dict[str, Any], idx: int) -> str:
    total = len(CATALOG)
    return f"<b>{door.get('name','Дверь')}</b>\nМодель {idx+1} из {total}"

async def show_or_update_carousel(cb_or_msg, state: FSMContext, idx: int):
    """
    Показываем/обновляем карусель с фото двери и кнопками.
    """
    idx = max(0, min(idx, len(CATALOG)-1))
    await state.update_data(carousel_idx=idx)
    door = CATALOG[idx]
    img_path = Path(door["image_png"])
    caption = door_caption(door, idx)
    kb = build_carousel_keyboard(idx)
    # Если это callback — пробуем редактировать; иначе — отправляем новое фото
    if isinstance(cb_or_msg, CallbackQuery):
        try:
            media = InputMediaPhoto(media=FSInputFile(str(img_path)), caption=caption, parse_mode="HTML")
            await cb_or_msg.message.edit_media(media=media, reply_markup=kb)
        except Exception:
            await cb_or_msg.message.answer_photo(photo=FSInputFile(str(img_path)), caption=caption, parse_mode="HTML", reply_markup=kb)
        await cb_or_msg.answer()
    else:
        await cb_or_msg.answer_photo(photo=FSInputFile(str(img_path)), caption=caption, parse_mode="HTML", reply_markup=kb)

# =========================== TELEGRAM BOT FLOW ===========================
@router.message(CommandStart())
async def start(m: Message, state: FSMContext):
    ok = await ensure_subscribed(m.from_user.id)
    if not ok:
        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="Подписаться на канал", url=f"https://t.me/{REQUIRED_BUILDER2112.strip('@')}")],
            [InlineKeyboardButton(text="✅ Проверить подписку", callback_data="check_sub")],
        ])
        await m.answer("Чтобы пользоваться ботом, подпишись на наш канал и нажми «Проверить подписку».", reply_markup=kb)
        return
    await state.clear()
    await m.answer("Пришлите <b>фотографию интерьера</b>. Мы подробно опишем сцену (без упоминаний дверей), затем выберете модель и цвет двери — и сгенерируем результат.", parse_mode="HTML")
    await state.set_state(Flow.waiting_foto)

@router.callback_query(F.data == "check_sub")
async def check_sub(cb: CallbackQuery, state: FSMContext):
    ok = await ensure_subscribed(cb.from_user.id)
    if not ok:
        await cb.answer("Ты ещё не подписан(а).", show_alert=True)
        return
    await cb.message.answer("Спасибо! Пришлите фото интерьера.")
    await state.set_state(Flow.waiting_foto)
    await cb.answer()

@router.message(Flow.waiting_foto, F.photo)
async def got_photo(m: Message, state: FSMContext):
    if not await ensure_subscribed(m.from_user.id):
        return
    workdir = Path("work") / str(m.from_user.id) / str(uuid.uuid4())
    img_path = workdir / "interior.jpg"
    await tg_download_photo(m, img_path)

    await state.set_state(Flow.describing)
    # Сообщения ожидания + индикатор печати
    await m.answer("⏳ Пожалуйста, ожидайте: происходит загрузка и анализ вашего изображения…")
    typing_stop = asyncio.Event()
    typing_task = asyncio.create_task(run_chat_action(m.chat.id, ChatAction.TYPING, typing_stop))

    try:
        english_desc, recommended_colors = await describe_scene_with_gemini(img_path)
    finally:
        typing_stop.set()
        try:
            await typing_task
        except Exception:
            pass

    # Покажем описание (можно разнести на несколько сообщений, если очень длинное)
    if english_desc:
        for chunk in textwrap.wrap(english_desc, 3500, replace_whitespace=False, drop_whitespace=False):
            await m.answer(truncate(chunk), parse_mode=None)

    await state.update_data(
        interior_path=str(img_path),
        interior_description_en=english_desc,
        recommended_colors=recommended_colors,
        carousel_idx=0
    )

    await m.answer("Выберите модель двери (листайте карусель):")
    await state.set_state(Flow.selecting_door)
    await show_or_update_carousel(m, state, idx=0)

@router.callback_query(Flow.selecting_door, F.data.startswith("carousel:"))
async def carousel_nav(cb: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    idx = current_catalog_index(data)
    action = cb.data.split(":")[1]
    if action == "prev":
        idx = (idx - 1) % len(CATALOG)
        await show_or_update_carousel(cb, state, idx)
    elif action == "next":
        idx = (idx + 1) % len(CATALOG)
        await show_or_update_carousel(cb, state, idx)
    elif action == "choose":
        # Зафиксировать текущую дверь
        door = CATALOG[idx]
        await state.update_data(door_id=str(door["id"]))
        # Подготовим список цветов: приоритет — из каталога, затем из Gemini, затем default_colors
        colors_catalog = door.get("colors") or []
        data = await state.get_data()
        colors_gemini = data.get("recommended_colors") or []
        merged: List[Dict[str, str]] = []
        if colors_catalog:
            merged.extend(colors_catalog)
        if colors_gemini:
            merged.extend([c for c in colors_gemini if c not in merged])
        # Если совсем пусто — fallback на default_colors
        if not merged:
            defaults = door.get("default_colors", []) or ["#FFFFFF", "#F3F0E6", "#D9D9D9", "#6B6B6B", "#2F5A3C", "#8B6A4E"]
            for hx in defaults:
                merged.append({"hex": hx, "name": hx})
        kb, descr = build_colors_keyboard_and_text(merged)
        await state.update_data(available_colors=merged)
        await cb.message.answer(
            f"Модель: <b>{door['name']}</b>\n\n{descr}\n\nВыберите цвет полотна и рамки (фурнитура НЕ перекрашивается):",
            parse_mode="HTML",
            reply_markup=kb
        )
        await cb.answer()
        await state.set_state(Flow.selecting_color)
    else:
        await cb.answer()

@router.callback_query(Flow.selecting_color, F.data.startswith("color_idx:"))
async def chose_color_from_list(cb: CallbackQuery, state: FSMContext):
    idx = int(cb.data.split(":")[1])
    data = await state.get_data()
    colors = data.get("available_colors", [])
    if 0 <= idx < len(colors):
        c = colors[idx]
        # Сформируем человеко-читаемую строку для промпта
        name = c.get("name", "").strip()
        ral = parse_color(c.get("ral", "").strip()) if c.get("ral") else ""
        hexv = parse_color(c.get("hex", "").strip()) if c.get("hex") else ""
        chosen_text = " ".join([v for v in [ral, name] if v]) or hexv or name or "neutral"
        await state.update_data(color_raw=c, color_text=chosen_text)
        await cb.answer()
        await generate_and_send(cb.message, state)
    else:
        await cb.answer("Неверный выбор цвета", show_alert=True)

@router.callback_query(Flow.selecting_color, F.data == "color:custom")
async def ask_custom_color(cb: CallbackQuery, state: FSMContext):
    await cb.message.answer("Напишите цвет: #HEX (например <code>#F3F0E6</code>), или <code>RAL 9010</code>, или простым словом (white, beige…).", parse_mode="HTML")
    await cb.answer()

@router.message(Flow.selecting_color)
async def typed_color(m: Message, state: FSMContext):
    color_user = parse_color(m.text or "")
    # Сохраним как есть для промпта, чтобы не терять RAL/имя
    if not color_user:
        await m.answer("Не удалось распознать цвет. Попробуйте снова: #HEX, RAL XXXX или название.")
        return
    await state.update_data(color_raw={"input": m.text.strip()}, color_text=color_user)
    await generate_and_send(m, state)

async def generate_and_send(m: Message, state: FSMContext):
    if not await ensure_subscribed(m.from_user.id):
        await m.answer("Сначала подпишитесь на канал и вернитесь с /start.")
        return
    await state.set_state(Flow.generating)
    data = await state.get_data()

    door_id = data.get("door_id")
    color_text = data.get("color_text", "")
    interior_en = data.get("interior_description_en", "")
    if not door_id or not color_text:
        await m.answer("Не выбраны дверь и/или цвет. Начните заново: /start")
        await state.clear()
        return

    # Находим файл двери
    try:
        door = next(d for d in CATALOG if str(d["id"]) == str(door_id))
    except StopIteration:
        await m.answer("Не удалось найти выбранную дверь. Начните заново: /start")
        await state.clear()
        return

    door_png = Path(door["image_png"])
    if not door_png.exists():
        await m.answer(f"Файл двери не найден: {door_png}")
        await state.clear()
        return

    # Дисклеймер + индикатор
    await m.answer("⏳ Пожалуйста, ожидайте: выполняется генерация вашего интерьера…\n\n<b>Важно!</b> Иногда изображения могут не соответствовать ожиданиям. "
                   "Попробуйте выбрать другую модель двери или другой интерьер. Цвета на изображении носят ориентировочный характер.", parse_mode="HTML")
    typing_stop = asyncio.Event()
    typing_task = asyncio.create_task(run_chat_action(m.chat.id, ChatAction.UPLOAD_PHOTO, typing_stop))

    try:
        img_bytes = await gemini_generate(door_png=door_png, color_text=color_text, interior_en=interior_en, aspect="3:4")
        try:
            file = BufferedInputFile(img_bytes, filename="result.png")
            await m.answer_photo(
                photo=file,
                caption=f"{door['name']} — выбранный цвет: {color_text}\nДверь по центру задней стены, полностью видима (ничем не закрыта)."
            )
        except Exception:
            tmp = Path("/tmp") / f"{uuid.uuid4().hex}.png"
            tmp.write_bytes(img_bytes)
            await m.answer_photo(photo=FSInputFile(str(tmp)), caption=f"{door['name']} — выбранный цвет: {color_text}")
    except Exception as e:
        print("GENERATION_ERROR:", repr(e))
        await m.answer("⚠️ Не удалось сгенерировать изображение. Проверьте ключи и попробуйте ещё раз.")
    finally:
        typing_stop.set()
        try:
            await typing_task
        except Exception:
            pass

    # Предложение продолжить
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔁 Выбрать другую дверь для этого интерьера", callback_data="again:door")],
        [InlineKeyboardButton(text="🆕 Начать заново с новым интерьером", callback_data="again:new")],
    ])
    await m.answer("Что дальше?", reply_markup=kb)
    await state.set_state(Flow.after_result)

@router.callback_query(Flow.after_result, F.data == "again:door")
async def again_door(cb: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    if not data.get("interior_description_en"):
        await cb.message.answer("Сессия описания интерьера не найдена. Запустите заново: /start")
        await state.clear()
        await cb.answer()
        return
    await cb.message.answer("Выберите другую модель двери (листайте карусель):")
    await state.set_state(Flow.selecting_door)
    await show_or_update_carousel(cb, state, idx=0)

@router.callback_query(Flow.after_result, F.data == "again:new")
async def again_new(cb: CallbackQuery, state: FSMContext):
    await state.clear()
    await cb.message.answer("Пришлите новое фото интерьера.")
    await state.set_state(Flow.waiting_foto)
    await cb.answer()

# Блокировка произвольных фото на этапе выбора двери/цвета
@router.message(Flow.selecting_door, F.photo)
async def reject_door_photo(m: Message):
    await m.answer("Пожалуйста, используйте карусель — вы не можете отправить своё фото двери. Листайте и выберите модель кнопкой «✅ Выбрать».")

# =========================== FASTAPI ===========================
app = FastAPI()

@app.get("/")
async def _health():
    return {"ok": True}

@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    update = await request.json()
    await dp.feed_webhook_update(bot, update)
    return {"ok": True}
