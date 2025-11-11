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
from aiogram.types import (Message, CallbackQuery, InlineKeyboardMarkup,
                           InlineKeyboardButton)
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext
from PIL import Image

# ------------ env ------------
load_dotenv()
BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
NANOBANANA_API_KEY = os.environ["NANOBANANA_API_KEY"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "")
REQUIRED_CHANNEL = os.environ.get("REQUIRED_CHANNEL", "@yourdoorshop")  # @username

bot = Bot(BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())
router = Router()
dp.include_router(router)

# ------------ catalog ------------
CATALOG = json.loads(Path("catalog.json").read_text(encoding="utf-8"))

# ------------ states ------------
class Flow(StatesGroup):
    waiting_photo = State()
    selecting_door = State()
    selecting_color = State()
    generating = State()

# ------------ helpers ------------
async def ensure_subscribed(user_id: int) -> bool:
    """
    Проверяем подписку на канал REQUIRED_CHANNEL.
    Канал должен быть публичным. Бот должен быть админом или хотя бы иметь доступ к участникам.
    """
    try:
        member = await bot.get_chat_member(REQUIRED_CHANNEL, user_id)
        status = getattr(member, "status", None)
        return status in ("member", "creator", "administrator")
    except Exception:
        return False

async def tg_download_photo(message: Message, dest: Path) -> Path:
    photo = max(message.photo, key=lambda p: p.file_size)
    f = await bot.get_file(photo.file_id)
    url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{f.file_path}"
    async with httpx.AsyncClient() as client:
        data = (await client.get(url)).content
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dest

def build_doors_keyboard(page: int = 0, per_page: int = 6) -> InlineKeyboardMarkup:
    start = page * per_page
    chunk = CATALOG[start:start+per_page]
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
    s = s.strip()
    if re.match(r"^#([0-9a-fA-F]{6})$", s): return s
    basic = {
        "white":"#FFFFFF","black":"#000000","beige":"#E6D8C3","cream":"#F3F0E6",
        "gray":"#BFBFBF","light gray":"#D9D9D9","dark gray":"#6B6B6B",
        "oak":"#D8C4A6","walnut":"#8B6A4E","green":"#2F5A3C","brown":"#6B4E2E"
    }
    return basic.get(s.lower(), s)  # допускаем "RAL 9010" / произвольные названия

# ------------ OpenAI Vision: JSON сцены ------------
async def describe_scene_with_openai(image_path: Path) -> Dict[str, Any]:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    system = "You are a precise scene describer. Return strict JSON only."
    schema_prompt = """
Analyze this interior photo and output a JSON object with:
{
  "style_keywords": [],
  "camera": { "type": "photo", "lens_mm": 35, "framing": "one_point_perspective", "view": "frontal" },
  "geometry": { "room_type": "", "ceiling_height_m": 2.7, "vanishing_lines": "towards center" },
  "surfaces": {
    "walls": { "color_hex": "#D7C8B6", "finish": "matte", "molding": "crown/baseboards/casings: yes" },
    "floor": { "material": "oak", "pattern": "herringbone", "finish": "matte" }
  },
  "lighting": {
    "key_light": "daylight from left",
    "practicals": ["wall sconce brass glass", "ceiling drum light"],
    "mood": "soft, warm, airy"
  },
  "materials_palette_hex": [],
  "metals": ["brushed brass","matte black"],
  "furniture_decor": [],
  "placement_hint": "door on end wall as unobstructed focal point"
}
Return JSON only.
"""
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": schema_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]}
        ],
        "temperature": 0.2
    }
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"].strip()
    try:
        return json.loads(txt)
    except Exception:
        return {"style_keywords": [], "materials_palette_hex": []}

# ------------ NaNobanana: генерация ------------

async def gemini_generate(
    base_image: Path, door_png: Path, color: str, scene: Dict[str, Any], aspect: str = "2:3"
) -> bytes:
    from google import genai
    from google.genai import types
    from PIL import Image
    import io, os

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    style_keywords = ", ".join(scene.get("style_keywords", []))
    walls_hex = scene.get("surfaces", {}).get("walls", {}).get("color_hex", "beige")
    floor = scene.get("surfaces", {}).get("floor", {})
    floor_material = floor.get("material", "oak")
    floor_pattern = floor.get("pattern", "herringbone")
    lighting = scene.get("lighting", {}).get("key_light", "soft daylight from left")
    palette = scene.get("materials_palette_hex", [])

    prompt = (
        "You are a photorealistic image editor. Compose a new image by integrating the DOOR IMAGE "
        "into the provided INTERIOR IMAGE with high realism.\n\n"
        "HARD REQUIREMENTS:\n"
        "- Place the door on the end wall as the main, unobstructed subject (frontal, one-point perspective, eye level, ~35mm look).\n"
        f"- Recolor only the door leaf to {color}; preserve panels, glazing, material grain, and hardware.\n"
        "- Keep verticals straight, realistic scale (~2.0–2.1 m door height). Do not add extra doors. No occlusion by decor.\n"
        "- Maintain the interior style, lighting, and palette exactly as in the base image.\n\n"
        "SCENE DETAILS (from analysis):\n"
        f"- Style: {style_keywords}\n"
        f"- Walls: {walls_hex} matte with white moldings.\n"
        f"- Floor: {floor_material} {floor_pattern} matte.\n"
        "- Metals: brushed brass, matte black.\n"
        f"- Lighting: {lighting} + warm sconce glow if present.\n"
        f"- Palette: {palette}\n\n"
        "QUALITY:\n"
        "Ultra-realistic PBR materials, soft daylight, accurate contact shadows, filmic tone mapping. "
        "Negative: glare, over-sharpening, plastic sheen, extra clutter, people, text, logos."
    )

    # Загружаем входные изображения
    interior_img = Image.open(base_image).convert("RGB")
    door_img = Image.open(door_png).convert("RGBA")  # PNG с альфой

    # Конфиг нового SDK: response_modalities в ВЕРХНЕМ регистре, ImageConfig для aspect_ratio
    cfg = types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(aspect_ratio=aspect),
    )

    resp = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[prompt, interior_img, door_img],
        config=cfg,
    )

    # Ищем картинку в ответе
    for part in resp.parts:
        if getattr(part, "inline_data", None):
            data = getattr(part.inline_data, "data", None)
            if data:
              return data
            try:
              out_img = part.as_image()
              buf = io.BytesIO()
              out_img.save(buf)
              return buf.getvalue()
            except Exception:
              continue
              
    raise RuntimeError("Gemini did not return an image bytes payload.")
  

async def nanobanana_generate(base_image: Path, door_png: Path, color: str,
                              scene: Dict[str,Any], seed: Optional[int] = None) -> bytes:
    """
    Универсальный вызов img2img. В личном кабинете NaNobanana посмотри точный URL и имена полей.
    Здесь — распространённый формат multipart (prompt + base_image + reference).
    """
    url = "https://api.nanobanana.ai/v1/image-to-image"  # замени на актуальный из их доков
    headers = {"Authorization": f"Bearer {NANOBANANA_API_KEY}"}
    prompt = f"""
Ultra-realistic interior photograph. Use the attached DOOR IMAGE as the main subject, centered on the end wall, unobstructed.
Recolor the door leaf to {color}, preserve panels/hardware/grain.
One-point perspective, frontal, eye level, 35mm.
Style: {', '.join(scene.get('style_keywords', []))}. Palette: {scene.get('materials_palette_hex', [])}.
Walls: {scene.get('surfaces',{}).get('walls',{}).get('color_hex','beige')} matte with white moldings.
Floor: {scene.get('surfaces',{}).get('floor',{}).get('material','oak')} {scene.get('surfaces',{}).get('floor',{}).get('pattern','herringbone')} matte.
Lighting: {scene.get('lighting',{}).get('key_light','soft daylight from left')} + warm brass sconce glow.
Decor to the sides only; do not occlude the door.
High detail, PBR, accurate contact shadows. Negative: glare, plastic sheen, extra doors, people, clutter.
"""
    files = {
        "prompt": (None, prompt),
        "mode": (None, "img2img"),
        "strength": (None, "0.55"),
        "guidance": (None, "3.5"),
        "seed": (None, str(seed or 42)),
        "base_image": ("interior.jpg", base_image.read_bytes(), "image/jpeg"),
        "reference_image_1": ("door.png", door_png.read_bytes(), "image/png"),
        "preserve_reference": (None, "door_exact")
    }
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(url, headers=headers, files=files)
        r.raise_for_status()
        return r.content

# ------------ Bot logic ------------
@router.message(CommandStart())
async def start(m: Message, state: FSMContext):
    ok = await ensure_subscribed(m.from_user.id)
    if not ok:
        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="Подписаться на канал", url=f"https://t.me/{REQUIRED_CHANNEL.strip('@')}")],
            [InlineKeyboardButton(text="✅ Проверить подписку", callback_data="check_sub")]
        ])
        await m.answer("Чтобы пользоваться ботом, подпишись на наш канал и нажми «Проверить подписку».", reply_markup=kb)
        return
    await state.clear()
    await m.answer("Пришли фото интерьера.")
    await state.set_state(Flow.waiting_photo)

@router.callback_query(F.data == "check_sub")
async def check_sub(cb: CallbackQuery, state: FSMContext):
    ok = await ensure_subscribed(cb.from_user.id)
    if not ok:
        await cb.answer("Ты ещё не подписан(а).", show_alert=True)
        return
    await cb.message.answer("Спасибо! Пришли фото интерьера.")
    await state.set_state(Flow.waiting_photo)
    await cb.answer()

@router.message(Flow.waiting_photo, F.photo)
async def got_photo(m: Message, state: FSMContext):
    # снова проверим подписку на случай отписки
    if not await ensure_subscribed(m.from_user.id):
        await m.answer("Сначала подпишись на канал и вернись с /start.")
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
    await cb.message.edit_reply_markup(build_doors_keyboard(page))
    await cb.answer()

@router.callback_query(Flow.selecting_door, F.data.startswith("door:"))
async def chose_door(cb: CallbackQuery, state: FSMContext):
    door_id = cb.data.split(":")[1]
    door = next(d for d in CATALOG if d["id"] == door_id)
    await state.update_data(door_id=door_id)
    palette = door.get("default_colors", [])
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=c, callback_data=f"color:{c}")] for c in palette[:6]
    ] + [[InlineKeyboardButton(text="Другой цвет…", callback_data="color:custom")]])
    await cb.message.answer(f"Модель: <b>{door['name']}</b>\nВыбери цвет или напиши свой (#HEX / RAL / слово).", reply_markup=kb)
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
    door = next(d for d in CATALOG if d["id"] == data["door_id"])
    door_png = Path(door["image_png"])
    color = data["color"]

    await m.answer("Генерирую…")

    scene = await describe_scene_with_openai(interior)
    img_bytes = await gemini_generate(interior, door_png, color, scene, aspect="2:3")

    await m.answer_photo(photo=img_bytes, caption=f"{door['name']} — цвет: {color}")
    await state.clear()
    await m.answer("Готово! Пришли новое фото, чтобы попробовать ещё.")

# ------------ FastAPI + webhook ------------
app = FastAPI()

@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    update = await request.json()
    await dp.feed_webhook_update(bot, update)
    return {"ok": True}
