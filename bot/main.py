from google import genai
from google.genai import types
import os, json, re, io, asyncio, uuid, base64
from pathlib import Path
from typing import Dict, Any, Optional

import httpx
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

load_dotenv()
BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
OPENAI_API_KEY = os.environ["ID"] if "ID" in os.environ else os.environ["OPENAI_API_KEY"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "")
REQUIRED_BUILDER2112 = os.getenv("REQUIRED_CHANNEL", "@yourdoorshop")
NANOBANANA_API_KEY = os.environ.get("NANOBANANA_API_KEY", "")

bot = Bot(BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())
router = Router()
dp.include_router(router)

CATALOG = json.loads(Path("catalog.json").read_text(encoding="utf-8"))

class Flow(StatesGroup):
    waiting_foto = State()
    selecting_door = State()
    selecting_color = State()
    generating = State()

async def ensure_subscribed(user_id: int) -> bool:
    try:
        member = await bot.get_chat_member(REQUIRED_BUILDER2112, user_id)
        status = getattr(member, "status", None)
        return status in ("member", "creator", "administrator")
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

def _json_sanitise(obj):
    if isinstance(obj, dict):
        return {k: _json_sanitise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_n for _n in map(_json_sanitise, obj)]
    return obj

async def describe_scene_with_openai(image_path: Path) -> Dict[str, Any]:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    system = "You are a precise interior scene describer. Return strict JSON only."
    schema_prompt = (
        "Analyze ONLY the non-door aspects of this interior photo and output a JSON object with EXACTLY this shape.\n"
        "Do NOT mention doors, door leaves, door frames, door hardware, openings, arches or thresholds. "
        "If the photo shows doors, IGNORE them completely. Describe walls as continuous planes; if there is an opening "
        "in the wall, do not mention it.\n\n"
        "Return JSON only:\n"
        "{\n"
        '  "style_keywords": [],\n'
        '  "camera": { "type": "photo", "lens_mm": 35, "framing": "one_point_perspective", "view": "frontal" },\n'
        '  "geo": { "room_type": "", "ceiling_height_m": 2.7, "vanishing_lines": "towards center" },\n'
        '  "surfaces": {\n'
        '    "walls": { "color_hex": "#D7C8B6", "finish": "matte", "molding": "crown/baseboards/casings: yes/no" },\n'
        '    "floor": { "material": "oak", "pattern": "herringbone", "finish": "matte" }\n'
        "  },\n"
        '  "lighting": {\n'
        '    "key_light": "daylight from left/right/front/back",\n'
        '    "practicals": ["wall sconce brass glass", "ceiling drum light"],\n'
        '    "mood": "soft, warm, airy"\n'
        "  },\n"
        '  "materials_palette_hex": [],\n'
        '  "metals": ["brushed brass","matte black"],\n'
        '  "furniture_decor": [],\n'
        '  "placement_hint": "treat back wall as a clean, doorless wall; final door will be inserted later"\n'
        "}"
    )
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": schema_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]}
        ],
        "temperature": 0.2,
    }
    async with hydra_client() as client:
        r = await client.post(url, headers=headers, json=_json_sanitise(payload))
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"].strip()
    try:
        data = json.loads(txt)
    except Exception:
        return {
            "style_keywords": [],
            "geo": {"room_type": "", "ceiling_height_m": 2.7, "vanishing_lines": "towards center"},
            "surfaces": {},
            "lighting": {},
            "materials_palette_hex": [],
            "metals": [],
            "furniture_decor": [],
            "placement_hint": "treat back wall as a clean, doorless wall; final door will be inserted later",
        }
    if "geometry" in data and "geo" not in data:
        data["geo"] = data.pop("geometry")
    data.setdefault("surfaces", {})
    data.setdefault("lighting", {})
    if "materials_palette_h\u200bex" in data and "materials_palette_hex" not in data:
        data["materials_palette_hex"] = data.pop("materials_palette_h\u200bex")
    data.setdefault("materials_palette_hex", [])
    data.setdefault("metals", data.get("metals", []))
    data.setdefault("style_keywords", data.get("style_keywords", []))
    data.setdefault("geo", {"room_type": "", "ceiling_height_m": 2.7, "vanishing_lines": "towards center"})
    return data

def hydra_client():
    return httpx.AsyncClient(timeout=300)

async def gemini_generate(
    door_png: Path,
    color: str,
    scene: Dict[str, Any],
    aspect: str = "2:3",
) -> bytes:
    client = genai.Client(api_key=GEMINI_API_KEY)
    style_words = ", ".join(scene.get("style_keywords", []))
    walls = scene.get("surfaces", {}).get("walls", {}) or {}
    floor = scene.get("surfaces", {}).get("floor", {}) or {}
    lighting = (scene.get("lighting", {}) or {}).get("key_light", "soft daylight from left")
    palette = scene.get("materials_palette_hex", [])
    prompt = f"""
Create an ultra-realistic interior photograph by RECONSTRUCTING the room from the following text only (no base photo is provided).
Then INSERT exactly ONE door leaf using the attached DOOR IMAGE. Do not create or mention any other doors/doorways.

ROOM (text spec; ignore any doors in the original photo):
- Style keywords: {style_words or "modern, calm, minimal"}
- Walls: {walls.get('color_hex', walls.get('color', '#e5e0e0'))} {walls.get('finish', 'matte')}, with simple white moldings/casings
- Floor: {floor.get('material','oak')} {floor.get('pattern','herringbone')} {floor.get('finish','matte')}
- Metals/accents: {", ".join(scene.get("metals", ["brushed brass","matte black"]))}
- Lighting: {lighting}; natural bounce, gentle soft shadows
- Palette accents: {", ".join(palette) if isinstance(palette, list) else ''}

DOOR (hard constraints):
- Use the attached DOOR IMAGE as the ONLY door. Keep its exact geometry (panels), proportions and hardware.
- Recolor the DOOR LEAF (panel surfaces only) to: {color}. Do NOT recolor metal hardware unless it is already colored in the source.
- Place the door centered on the back wall (one-point perspective, eye-level, ~35 mm).
- Do NOT render any other doors/arches/openings. Ensure proper wall thickness and contact shadows at threshold.

QUALITY:
- Photorealistic PBR shading; correct perspective; no HDR halos; no over-sharpening; no extra text/people/clutter.
"""
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
    if hasattr(resp, "parts"):
        for part in resp.parts:
            if hasattr(part, "inline_data") and getattr(part, "inline_data", None):
                data = part.inline_data.data
                if data:
                    return data
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
                if getattr(p, "inline_data", None):
                    data = p.inline_data.data
                    if data:
                        return data
    raise RuntimeError("Gemini did not return an image payload")

async def nanobanana_generate(base_image: Path, door_png: Path, color: str,
                              scene: Dict[str,Any], seed: Optional[int] = None) -> bytes:
    url = "https://api.nanobanana.ai/v1/image-to-image"
    headers = {"Authorization": f"Bearer {NANOBANANA_API_KEY}"}
    prompt = f"""
Ultra-realistic interior photograph. Use the attached DOOR IMAGE as the main subject, centered on the back wall.
Recolor the door leaf to {color}, preserve panels/hardware/grain. One-point perspective, eye level.
Style: {', '.join(scene.get('style_keywords', []))}. Palette: {scene.get('materials_palette_hex', [])}.
Walls: {scene.get('surfaces',{}).get('walls',{}).get('color_hex','beige')} matte with white moldings.
Floor: {scene.get('surfaces',{}).get('floor',{}).get('material','oak')} {scene.get('surfaces',{}).get('floor',{}).get('pattern','herringbone')} matte.
Lighting: {scene.get('lighting',{}).get('key_light','soft daylight from left')}.
No extra doors/arches; decor kept to the sides; accurate contact shadows.
"""
    files = {
        "prompt": (None, prompt),
        "mode": (None, "img2img"),
        "strength": (None, "0.55"),
        "guidance": (None, "3.5"),
        "seed": (None, str(seed or 42)),
        "base_image": ("interior.jpg", base_image.read_bytes(), "image/jpeg"),
        "reference_image_1": ("door.png", door_png.read_bytes(), "image/png"),
        "preserve_reference": (None, "door_exact"),
    }
    async with hydra_client() as client:
        r = await client.post(url, files=files, headers=headers)
        r.raise_for_status()
        return r.content

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
    await m.answer("Пришли фото интерьера.")
    await state.set_state(Flow.waiting_foto)

version_decorator = None

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
    await state.update_data(interior_path=str(img_path))
    await m.answer("Фото получено. Выбери модель двери:", reply_markup=build_doors_keyboard(0))
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
    palette = door.get("default_colors", [])
    kb_rows = [[InlineKeyboardButton(text=c, callback_data=f"color:{c}")] for c in (palette[:6] if isinstance(palette, list) else [])]
    kb_rows.append([InlineKeyboardButton(text="Другой цвет…", callback_data="color:custom")])
    await cb.message.answer(f"Модель: <b>{door['name']}</b>\nВыбери цвет или напиши свой (#HEX / RAL / слово).",
                            reply_markup=InlineKeyboardMarkup(inline_keyboard=kb_rows))
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
    interior = Path(data["interior_path"])
    door_id = data.get("door_id")
    door = next(d for d in CATALOG if str(d["id"]) == str(door_id))
    door_png = Path(door["image_png"])
    color = parse_color(data.get("color", ""))

    if not door_png.exists():
        await m.answer(f"Файл двери не найден: {door_png}")
        await state.clear()
        return

    await m.answer("Генерирую…")
    try:
        scene = await describe_scene_with_openai(interior)
        img_bytes = await gemini_generate(door_png=door_png, color=color, scene=scene, aspect="2:3")
        try:
            file = BufferedInputFile(img_bytes, filename="result.png")
            await m.answer_photo(photo=file, caption=f"{door['name']} — цвет: {color}")
        except Exception:
            tmp = Path("/tmp") / f"{uuid.uuid4().hex}.png"
            tmp.write_bytes(img_bytes)
            await m.answer_photo(photo=FSInputFile(str(tmp)), caption=f"{door['name']} — цвет: {color}")
    except Exception as e:
        print("GENERATION_ERROR:", repr(e))
        await m.answer("⚠️ Не удалось сгенерировать изображение. Проверьте ключи и попробуйте ещё раз.")
    finally:
        await state.clear()

app = FastAPI()

@app.get("/")
async def _health():
    return {"ok": True}

@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    update = await request.json()
    await dp.feed_webhook_update(bot, update)
    return {"ok": True}
