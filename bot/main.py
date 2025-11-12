from google import genai
from google.genai import types
import os, json, io, uuid, re
from pathlib import Path
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import CommandStart
from aiogram.types import (
    Message,
    CallbackQuery,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    BufferedInputFile,
    FSInputFile,
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
    selecting_door = State()
    selecting_color = State()
    generating = State()

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

def build_doors_keyboard(page: int = 0, per_page: int = 6) -> InlineKeyboardMarkup:
    start = page * per_page
    chunk = CATALOG[start:start + per_page]
    rows = [[InlineKeyboardButton(text=d["name"], callback_data=f"door:{d['id']}")] for d in chunk]
    nav = []
    if start > 0:
        nav.append(InlineKeyboardButton(text="◀ Назад", callback_data=f"page:{page-1}"))
    if start + per_page < len(CATALOG):
        nav.append(InlineKeyboardButton(text="Вперёд ▶", callback_data=f"page:{page+1}"))
    if nav:
        rows.append(nav)
    return InlineKeyboardMarkup(inline_keyboard=rows)

def parse_color(s: str) -> str:
    s = (s or "").strip()
    if re.match(r"^#([0-9A-Fa-f]{6})$", s):
        return s
    basic = {
        "white": "#FFFFFF", "black": "#000000", "beige": "#E6D8C3", "cream": "#F3F0E6",
        "gray": "#BFBFBF", "light gray": "#D9D9D9", "dark gray": "#6B6B6B",
        "oak": "#D8C4A6", "walnut": "#8B6A4E", "green": "#2F5A3C", "brown": "#6B4E2E"
    }
    return basic.get(s.lower(), s)

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
async def describe_scene_with_gemini(image_path: Path) -> Dict[str, Any]:
    """
    Анализируем ТОЛЬКО не-дверные аспекты фото.
    Возвращаем строгий JSON по согласованной схеме.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)

    schema_prompt = (
        "You are a precise interior scene describer. "
        "Return STRICT JSON ONLY (no prose, no markdown). "
        "Analyze ONLY non-door aspects of the photo. "
        "Do NOT mention doors, door leaves, door frames, door hardware, openings, arches or thresholds. "
        "If doors/openings are visible, IGNORE them completely. "
        "Treat every wall as a continuous plane; if an opening exists, do not mention it.\n\n"
        "Return JSON with EXACTLY these keys:\n"
        "{\n"
        '  \"style_keywords\": [],\n'
        '  \"camera\": { \"type\": \"photo\", \"lens_mm\": 35, \"framing\": \"one_point|two_point\", \"view\": \"frontal|slight_angle\" },\n'
        '  \"geo\": { \"room_type\": \"\", \"ceiling_height_m\": 2.7, \"vanishing_lines\": \"towards center\" },\n'
        '  \"surfaces\": {\n'
        '    \"walls\": { \"color_hex\": \"#D7C8B6\", \"finish\": \"matte|eggshell|satin\", \"molding\": \"crown/baseboards/casings: yes|no\" },\n'
        '    \"floor\": { \"material\": \"oak|tile|stone|concrete\", \"pattern\": \"planks|herringbone|grid\", \"finish\": \"matte|satin|gloss\" }\n'
        "  },\n"
        '  \"lighting\": {\n'
        '    \"key_light\": \"daylight from left/right/front/back\", \"mood\": \"soft|crisp|dramatic\", \"ct\": 3500\n'
        "  },\n"
        '  \"materials_palette_hex\": [],\n'
        '  \"metals\": [\"brushed brass\",\"matte black\"],\n'
        '  \"furniture_decor\": [\n'
        '     {\"item\":\"dresser\",\"count\":1},{\"item\":\"nightstand\",\"count\":1},{\"item\":\"bed\",\"count\":1}\n'
        "  ],\n"
        '  \"placement_hint\": \"treat back wall as a clean, doorless wall; final door will be inserted later\"\n'
        "}"
    )

    img = Image.open(image_path).convert("RGB")

    resp = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[schema_prompt, img],
        config=types.GenerateContentConfig(temperature=0.1),
    )

    txt = _resp_text(resp).strip()
    # Пытаемся распарсить JSON
    try:
        data = json.loads(txt)
    except Exception:
        # fallback — пустая заготовка
        data = {
            "style_keywords": [],
            "camera": {"type": "photo", "lens_mm": 40, "framing": "two_point", "view": "slight_angle"},
            "geo": {"room_type": "", "ceiling_height_m": 2.7, "vanishing_lines": "towards center"},
            "surfaces": {},
            "lighting": {},
            "materials_palette_hex": [],
            "metals": [],
            "furniture_decor": [],
            "placement_hint": "treat back wall as a clean, doorless wall; final door will be inserted later"
        }

    # Нормализация ключей
    if "geometry" in data and "geo" not in data:
        data["geo"] = data.pop("geometry")
    data.setdefault("surfaces", {})
    data.setdefault("lighting", {})
    if "materials_palette_h\u200bex" in data and "materials_palette_hex" not in data:
        data["materials_palette_hex"] = data.pop("materials_palette_h\u200bex")
    data.setdefault("materials_palette_hex", [])
    data.setdefault("metals", data.get("metals", []))
    data.setdefault("style_keywords", data.get("style_keywords", []))
    data.setdefault("camera", data.get("camera", {"type":"photo","lens_mm":40,"framing":"two_point","view":"slight_angle"}))
    data.setdefault("geo", data.get("geo", {"room_type": "", "ceiling_height_m": 2.7, "vanishing_lines": "towards center"}))
    return data

# =========================== 2) ГЕНЕРАЦИЯ КАДРА (Gemini 2.5 Flash Image) ===========================
def build_generation_prompt(scene: Dict[str, Any], door_color: str) -> str:
    """
    Строим единый промпт для gemini-2.5-flash-image:
    - реконструируем комнату ТОЛЬКО по тексту (без фото);
    - затем вставляем РОВНО одну дверь из приложенной картинки;
    - строгие ограничения на дверь (геометрия/фурнитура, цвет полотна, центр задней стены, запрет перекрытия).
    """

    style_words = ", ".join(scene.get("style_keywords", []) or []) or "mid-century modern, calm, cozy"
    cam = scene.get("camera", {}) or {}
    framing = cam.get("framing", "two_point")
    view = cam.get("view", "slight_angle")
    lens = cam.get("lens_mm", 40)

    walls = (scene.get("surfaces", {}) or {}).get("walls", {}) or {}
    floor = (scene.get("surfaces", {}) or {}).get("floor", {}) or {}
    lighting = scene.get("lighting", {}) or {}
    metals = ", ".join(scene.get("metals", []) or ["brushed brass"])

    palette = scene.get("materials_palette_hex", [])
    palette_str = ", ".join(palette) if isinstance(palette, list) else ""

    door_color_hex = parse_color(door_color) or "#FFFFFF"

    return f"""
Create an ULTRA-REALISTIC interior photograph by RECONSTRUCTING the room from the following text ONLY (no base photo is provided).
Then INSERT exactly ONE door leaf using the attached DOOR IMAGE. Do not create or mention any other doors/doorways.

ROOM (derived from the user's photo; ignore any doors in that photo):
- Style keywords: {style_words}
- Camera & framing: {framing} perspective, {view} view; vertical 3:4; ~{lens} mm; camera height ~150 cm; slight DoF.
- Walls: color {walls.get('color_hex', '#e5e0e0')} with {walls.get('finish','matte')} finish; moldings: {walls.get('molding','minimal/none')}.
- Floor: {floor.get('material','oak')} {floor.get('pattern','planks')} with {floor.get('finish','satin')} finish.
- Lighting: key = {lighting.get('key_light','soft diffuse daylight from left')}; mood = {lighting.get('mood','soft')}; CT ≈ {lighting.get('ct', 3800)} K; balanced exposure; no HDR/bloom; physically plausible GI.
- Metals/accents: {metals}
- Palette accents: {palette_str}

DOOR (hard constraints):
- Use the attached DOOR IMAGE as the ONLY door. Keep its exact geometry (panel layout), proportions and hardware.
- Recolor the DOOR LEAF (panel surfaces only) to: {door_color_hex}. Do NOT recolor metal hardware.
- Place the door centered on the BACK WALL (one-point alignment relative to that wall), camera eye-level ~35–40 mm equivalence.
- The door must be FULLY VISIBLE and UNOBSTRUCTED: do not block or partially hide it with any furniture, decor, plants, textiles, or other objects.
- Do NOT render any other doors/arches/openings. Ensure proper wall thickness and realistic contact shadows at the threshold and trim.

QUALITY:
- Photorealistic PBR shading; correct perspective; clean global illumination; accurate color management; minimal noise.
- No HDR halos, no over-sharpening, no extra text/people/clutter.
"""

async def gemini_generate(door_png: Path, color: str, scene: Dict[str, Any], aspect: str = "3:4") -> bytes:
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = build_generation_prompt(scene=scene, door_color=color)
    img = Image.open(door_png).convert("RGBA")

    cfg = types.GenerateContentConfig(
        response_modalities=["Image"],
        image_config=types.ImageConfig(aspect_ratio=aspect),
    )
    resp = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[prompt, img],
        config=cfg,
    )
    return _resp_image_bytes(resp)

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
    await m.answer("Пришли фото интерьера. Мы опишем сцену (без учёта дверей), затем выберешь модель и цвет двери — и сгенерируем результат.")
    await state.set_state(Flow.waiting_foto)

@router.callback_query(F.data == "check_sub")
async def check_sub(cb: CallbackQuery, state: FSMContext):
    ok = await ensure_subscribed(cb.from_user.id)
    if not ok:
        await cb.answer("Ты ещё не подписан(а).", show_alert=True)
        return
    await cb.message.answer("Спасибо! Пришли фото интерьера.")
    await state.set_state(Flow.waiting_foto)
    await cb.answer()

@router.message(Flow.waiting_foto, F.photo)
async def got_photo(m: Message, state: FSMContext):
    if not await ensure_subscribed(m.from_user.id):
        return
    workdir = Path("work") / str(m.from_user.id) / str(uuid.uuid4())
    img_path = workdir / "interior.jpg"
    await tg_download_photo(m, img_path)

    # 1) Описываем сцену через Gemini 2.5 Pro
    await m.answer("Анализирую фото (игнорирую двери)…")
    scene = await describe_scene_with_gemini(img_path)

    await state.update_data(interior_path=str(img_path), scene=scene)
    await m.answer("Фото проанализировано. Выбери модель двери:", reply_markup=build_doors_keyboard(0))
    await state.set_state(Flow.selecting_door)

@router.callback_query(Flow.selecting_door, F.data.startswith("page:"))
async def paginate(cb: CallbackQuery):
    page = int(cb.data.split(":")[1])
    await cb.message.edit_reply_markup(reply_markup=build_doors_keyboard(page))
    await cb.answer()

@router.callback_query(Flow.selecting_door, F.data.startswith("door:"))
async def chose_door(cb: CallbackQuery, state: FSMContext):
    door_id = cb.data.split(":")[1]
    door = next(d for d in CATALOG if str(d["id"]) == str(door_id))
    await state.update_data(door_id=door_id)

    # Кнопки выбора цвета
    palette = door.get("default_colors", []) or ["#FFFFFF", "#F3F0E6", "#D9D9D9", "#6B6B6B", "#2F5A3C", "#8B6A4E"]
    kb_rows = [[InlineKeyboardButton(text=c, callback_data=f"color:{c}")] for c in palette[:6]]
    kb_rows.append([InlineKeyboardButton(text="Другой цвет…", callback_data="color:custom")])

    await cb.message.answer(
        f"Модель: <b>{door['name']}</b>\nВыбери цвет полотна (перекрашиваются только панели; фурнитура остаётся как в источнике).",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=kb_rows)
    )
    await cb.answer()
    await state.set_state(Flow.selecting_color)

@router.callback_query(Flow.selecting_color, F.data.startswith("color:"))
async def chose_color(cb: CallbackQuery, state: FSMContext):
    val = cb.data.split(":")[1]
    if val == "custom":
        await cb.message.answer("Напиши цвет: #HEX (например #F3F0E6), или RAL 9010, или слово (white, beige…).")
        await cb.answer()
        return
    await state.update_data(color=val)
    await cb.answer()
    await generate_and_send(cb.message, state)

@router.message(Flow.selecting_color)
async def typed_color(m: Message, state: FSMContext):
    color = parse_color(m.text)
    await state.update_data(color=color)
    await generate_and_send(m, state)

async def generate_and_send(m: Message, state: FSMContext):
    if not await ensure_subscribed(m.from_user.id):
        await m.answer("Сначала подпишись на канал и вернись с /start.")
        return
    await state.set_state(Flow.generating)
    data = await state.get_data()

    door_id = data.get("door_id")
    color = parse_color(data.get("color", ""))
    scene = data.get("scene", {})
    if not door_id or not color:
        await m.answer("Не выбраны дверь и/или цвет. Начни заново: /start")
        await state.clear()
        return

    door = next(d for d in CATALOG if str(d["id"]) == str(door_id))
    door_png = Path(door["image_png"])
    if not door_png.exists():
        await m.answer(f"Файл двери не найден: {door_png}")
        await state.clear()
        return

    await m.answer("Генерирую изображение…")
    try:
        img_bytes = await gemini_generate(door_png=door_png, color=color, scene=scene, aspect="3:4")
        try:
            file = BufferedInputFile(img_bytes, filename="result.png")
            await m.answer_photo(photo=file, caption=f"{door['name']} — цвет полотна: {color}\nДверь по центру задней стены, полностью видима (ничем не закрыта).")
        except Exception:
            tmp = Path("/tmp") / f"{uuid.uuid4().hex}.png"
            tmp.write_bytes(img_bytes)
            await m.answer_photo(photo=FSInputFile(str(tmp)), caption=f"{door['name']} — цвет полотна: {color}")
    except Exception as e:
        print("GENERATION_ERROR:", repr(e))
        await m.answer("⚠️ Не удалось сгенерировать изображение. Проверьте ключи и попробуйте ещё раз.")
    finally:
        await state.clear()

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
