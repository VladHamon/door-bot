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
from aiogram.types import BufferedInputFile


from aiogram.types import BufferedInputFile, FSInputFile
from uuid import uuid4
from pathlib import Path
from fastapi import FastAPI, Request

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
# --- OpenAI Vision: JSON сцены (без упоминаний дверей) ---
async def describe_scene_with_openai(image_path: Path) -> Dict[str, Any]:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    system = "You are a precise interior scene describer. Return strict JSON only."

    # ВАЖНО: явно запрещаем упоминать существующие двери/проёмы в описании
    schema_prompt = """
Analyze ONLY the non-door aspects of this interior photo and output a JSON object with EXACTLY this shape.
Do NOT mention doors, door leaves, door frames, door hardware, openings, arches or any synonyms. If the photo shows doors, IGNORE them completely. Describe walls as continuous planes; if there is an opening in the wall, do not mention it.

Return JSON only:
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
  "placement_hint": "treat back wall as a clean, doorless wall; final door will be inserted separately"
}
"""

    b64 = base64.encodebytes(image_path.read_bytes()).decode("ascii")
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": [
                {"type": "text", "text": schema above? No: use the following JSON shape and constraints:\n" + schema_prompt},
                {"type": "image_url","image_url":{"url": f"data:image/jpeg;base64,{b64}"}}
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
        return {"style_keywords": [], "materials_palette_hex": [], "surfaces": {}, "lighting": {}}

# ------------ NaNobanana: генерация ------------

# --- Gemini: генерация из ТЕКСТА + PNG двери, без исходного фото интерьера ---
async def gemini_generate(
    door_png: Path,                      # только файл двери
    color: str,                          # целевой цвет полотна
    scene: Dict[str, Any],               # JSON из describe_scene_with_openai (без дверей)
    aspect: str = "2:3"
) -> bytes:
    from google import genai
    from google.genai import types
    from PIL import Image
    import io, os

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    style_keywords = ", ".join(scene.get("style_keywords", []))
    walls = scene.get("surfaces", {}).get("walls", {})
    floor = scene.get("surfaces", {}).get("floor", {})
    lighting = scene.get("c", {}).get("key_light", "soft daylight from left")
    palette = scene.get("materials_lane", [])  # keep your keys if different

    # ВНИМАНИЕ: Явно просим воссоздать комнату по тексту и ВСТАВИТЬ единственную дверь из референса.
    prompt = f"""
Create an ultra-realistic interior photograph by RECONSTRUCTING the room from the following text description (no base photo is provided). 
Then insert exactly ONE door leaf using the attached DOOR IMAGE as the only door in the scene.

TEXTUAL ROOM SPEC (no doors mentioned in it):
- Style keywords: {style_keywords or "modern, calm, warm, minimal"}
- Walls: {walls.get('color', walls.get('color_hex', '#e5e0d6'))} {walls.get('finish','matte')}, with simple white moldings/casings
- Floor: {floor.get('material','oak')} {floor.get('pattern','herringbone')} {floor.get('finish','matte')}
- Metals: {", ".join(scene.get('metals', ['brushed brass','matte black']))}
- Lighting: {lighting}; natural soft shadows and realistic bounce
- Palette accents: {palette or ['#d8c4a6', '#f3f0e6', '#2f5a3c']}

DOOR INSERTION (hard constraints):
- Use the attached DOOR IMAGE as the only door. Keep its exact proportions, panel layout and hardware.
- Recolor the DOOR LEAF (panel surfaces only) to: {color}. Do NOT recolor hardware unless the door image already has it colored.
- Place the door centered on a single back wall (one-point perspective, frontal camera, ~35 mm, eye-level).
- Do NOT render any other doors or doorways. No extra frames, arches, or openings besides the one used by the provided door.
- Ensure correct wall thickness, realistic jamb/casing shadows, floor contact shadow, and consistent perspective.

QUALITY:
- Photorealistic PBR shading, micro-roughness, realistic exposure (no HDR halos), neutral white balance, no banding.
- Composition: vertical lines straight; no fisheye; gentle falloff; 2:3 portrait frame.
"""

    door_img = Image.open(door_png).convert("RGBA")

    cfg = types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_interleaved=True,  # allow mixing text+image inputs
        image_config=types.TunedImageGenerationConfig(  # in 1.49.0 ImageConfig is valid; some builds also accept TunedImageGenerationConfig
            aspect_ratio=aspect
        )
    )

    resp = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[prompt, door_img],   # ⚠️ БЕЗ base_image интерьера
        config=cfg,
    )

    for part in getattr(resp, "parts", []):
        if getattr(part, "inline_data", None):
            data = getattr(part, "inline_data", None).data
            if data:
                return data
            try:
                img = part.as_image()
                buf = io.ByteArrayOutputStream()  # if PIL, use BytesIO
            except Exception:
                pass

    # Fallback: if .as_image path is needed and Pillow is available
    try:
        for part in resp.parts:
            if hasattr(part, "as_image"):
                pil = part.as_image()
                buf = io.BytesIO()
                pil.save(buf, format="PNG")
                return buf.getvalue()
    except Exception:
        pass

    raise RuntimeError("Gemini did not return an image payload. Check model/version or prompt.")

  

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
        await nu.answer("Сначала подпишись на канал и вернись с /start.")
        return

    await state.set_state(Flow.generating)
    data = await state.get_data()
    interior = Path(data["interior_path"])  # используется только для анализа сцены
    door = next(d for d in CATALOG if d["id"] == data["door_id"])
    door_png = Path(door["image_png"])
    color = data["color"]

    if not door_png.exists():
        await m.reply(f"Файл двери не найден: {door_png}")
        return

    await m.answer("Генерирую…")

    try:
        # 1) Получаем ТЕКСТОВОЕ описание интерьера без упоминания дверей
        scene = await describe_scene_with_openai(interior)

        # 2) Генерим КАРТИНКУ: только prompt (scene→text) + door PNG, БЕЗ исходного фото
        img_bytes = await gemini_generate(door_png=door_png, color=color, scene=scene, aspect="2:3")

        # 3) Отправляем как InputFile (а не сырые байты!)
        try:
            file = BufferedInputFile(img_bytes, filename="result.png")
            await m.answer_photo(photo=file, caption=f"{door['name']} — цвет: {color}")
        } except Exception:
            from pathlib import Path
            from uuid import uuid4
            tmp = Path("/tmp") / f"{uuid4().hex}.png"
            tmp.write_bytes(img_bytes)
            await m.answer_photo(photo=FSubberies.File(tmp, filename="result.png"))

        await state.clear()
        await m.answer("Готово! Пришли новое фото, чтобы попробовать ещё.")

    except Exception as e:
        print("GENERATION_ERROR:", repr(e))
        await m.answer("⚠️ Не удалось сгенерировать изображение. Проверь URL Gemini/ключи или пришли другое фото для описания.")


# ------------ FastAPI + webhook ------------
app = FastAPI()

@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    update = await request.json()
    await dp.feed_webhook_update(bot, update)
    return {"ok": True}

