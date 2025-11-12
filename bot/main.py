from google import genai
from google.genai import types
import os, json, io, uuid
from pathlib import Path

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

# -------------------- ENV --------------------
load_dotenv()
BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "")
REQUIRED_BUILDER2112 = os.getenv("REQUIRED_CHANNEL", "@yourdoorshop")

# -------------------- TELEGRAM CORE --------------------
bot = Bot(BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())
router = Router()
dp.include_router(router)

# Каталог дверей: [{"id": "...", "name": "...", "image_png": "...", ...}, ...]
CATALOG = json.loads(Path("catalog.json").read_text(encoding="utf-8"))

# -------------------- FSM --------------------
class Flow(StatesGroup):
    selecting_door = State()
    generating = State()

# -------------------- UTILS --------------------
async def ensure_subscribed(user_id: int) -> bool:
    try:
        member = await bot.get_chat_member(REQUIRED_BUILDER2112, user_id)
        return getattr(member, "status", None) in ("member", "creator", "administrator")
    except Exception:
        return False

def build_doors_keyboard(page: int = 0, per_page: int = 6) -> InlineKeyboardMarkup:
    start = page * per_page
    chunk = CATALOG[start : start + per_page]
    rows = [[InlineKeyboardButton(text=d["name"], callback_data=f"door:{d['id']}")] for d in chunk]
    nav = []
    if start > 0:
        nav.append(InlineKeyboardButton(text="◀ Назад", callback_data=f"page:{page-1}"))
    if start + per_page < len(CATALOG):
        nav.append(InlineKeyboardButton(text="Вперёд ▶", callback_data=f"page:{page+1}"))
    if nav:
        rows.append(nav)
    return InlineKeyboardMarkup(inline_keyboard=rows)

# -------------------- PROMPT --------------------
def build_gemini_prompt() -> str:
    """
    Жёстко зафиксированный промпт под gemini-2.5-flash-image,
    соответствует вашему ТЗ + запрет перекрывать дверь.
    """
    return """Create an ultra-realistic interior photograph in Mid-Century Modern style.
Reconstruct the entire room ONLY from this text (no base photo is provided).

Then INSERT exactly ONE door leaf using the ATTACHED DOOR IMAGE:
— The attached door is the ONLY door allowed in the scene.
— Preserve its exact geometry, proportions, and hardware.
— Recolor the DOOR LEAF panels ONLY to pure white (#FFFFFF). Do NOT recolor or modify metal hardware.
— Place the door centered on the BACK WALL, one-point placement for the door (camera eye level ~35–40 mm).
— The door must be FULLY VISIBLE and UNOBSTRUCTED: do not block or partially hide it with any furniture, decor, plants, textiles, or other objects.
— Do NOT create, hint, or show any other doors, openings, arches, or thresholds.
— Ensure correct wall thickness at the opening and realistic contact shadows at the threshold/trim.

ROOM DESCRIPTION (follow precisely):
[CAMERA & FRAMING]
• Perspective: two-point; slight angle from the left; vertical 3:4; ~40 mm lens; camera height ~150 cm.
• Focus plane: on the nightstand and edge of the bed. Slight DOF; dresser and back wall remain sharp.
• Framing: include dresser (left, foreground), nightstand (center-left), bed (right), and a clear centered view of the door on the back wall.

[LIGHTING]
• Soft, diffuse daylight from a large source on the left (out of frame).
• Very soft, subtle shadows cast to the right.
• Ambient fill: neutral; gentle bounce from white ceiling and bedding.
• Color temperature: ~3500–4000 K (warm-neutral). Balanced exposure. No bloom, no HDR look, no stylization.

[COLOR PALETTE]
• Walls: muted olive green #69694A
• Trim/Crown: off-white #F5F4F0
• Ceiling: white #FFFFFF
• Bedding: crisp white #FFFFFF
• Wood: warm walnut #6B4933
• Floor (oak): #8A6240
• Metals: brushed brass/gold #B08D57
• Throw: textured grey #6A6761
• Cushion: dark tan #745A3D
• Rug: cream #E3DACE, grey #9A9A9A, gold #AF8C5A
• Greenery: muted green #5E6B4A

[ARCHITECTURE & ENVELOPE]
• Walls: smooth eggshell paint (#69694A). Left wall is a continuous plane meeting the back wall at 90°.
• Ceiling: flat white (#FFFFFF); simple stepped crown in #F5F4F0.
• Baseboards/trim: none visible (except door trim as needed).
• Floor: hardwood planks, medium-tone oak #8A6240, 3–4 inch width, satin finish.
• No windows visible.

[RUG]
• Low-pile wool, rectangular; woven texture; bound edge.
• Placement: under nightstand and partially under bed, perpendicular to bed.

[FURNITURE]
• Dresser (left/foreground): ~140×45×75 cm; walnut veneer #6B4933 (satin); mid-century; tapered angled legs;
  lower section: 3 large drawers w/ long horizontal wood pulls; top: 3 small drawers (2 left, 1 right) w/ small round inset pulls; pristine.
• Nightstand (back wall, left of bed): ~55×40×60 cm; walnut veneer; one drawer w/ two small round inset pulls, open shelf; tapered legs; pristine.
• Bed (back wall, right): queen (~160 cm wide); walnut platform, low rectangular headboard, tapered legs; pristine.

[DECOR & OBJECTS]
• Wall panel: one large vertical panel with arched/curved top, cream #EAE8DE, centered above the dresser.
• Wall sconce: cone shade, olive #5A5A3C; mounted at center of the panel; OFF.
• On dresser (left→right): 1 large light-grey vase #B8B4AA with 3–4 leafy stems #5E6B4A; 1 small clear glass bud vase;
  1 stack of 2–3 books (one dark red spine); 1 small brass sphere #B08D57 (≈6–8 cm).
• On nightstand (left→right): 1 small black vase #3A3A3A; 1 medium matte light-grey vase #C4C4C4 with 2 leafy stems; 1 small stack of 2 white books.

[TEXTILES]
• Duvet & sheets: crisp white cotton; folded down; soft, realistic folds.
• Sleep pillows: 2 white.
• Decorative pillow: 1 rectangular, dark tan #745A3D (velvet/chenille).
• Throw: chunky knit medium-grey #6A6761, draped over right corner of bed.

[MATERIAL & SHADER NOTES]
• PBR realism; correct micro-roughness and IOR.
• Wood: satin sheen, clear grain (horizontal on drawer fronts, vertical on legs/frame).
• Paint: eggshell low gloss.
• Metal: brushed brass (low gloss).
• Ceramics: matte/satin.
• Fabric: visible weave/pile; natural drape.
• Lighting: physically plausible global illumination. No HDR halos, no over-sharpening, no bloom.

[SCALE]
• Ceiling ~2.7 m; dresser ~140 cm W; nightstand ~55 cm W; headboard ~80–85 cm H.

[NEGATIVE & CONSTRAINTS]
• Absolutely NO other doors/arches/openings/thresholds.
• Do not invent extra rooms or spaces.
• No people or pets.
• Keep wall continuity; treat left wall as solid plane.
• The door must remain completely visible with NO occlusion by any item.

[RENDER QUALITY]
• Photorealistic output; crisp textures; minimal noise; accurate color management; natural contrast.
"""

# -------------------- GEMINI --------------------
def read_image_rgba(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGBA")
    return img

def as_image_bytes_from_response(resp) -> bytes:
    # Универсальный «сборщик» из ответа Gemini
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
    raise RuntimeError("Gemini did not return an image payload")

def gemini_generate(door_png: Path, aspect: str = "3:4") -> bytes:
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = build_gemini_prompt()
    door_img = read_image_rgba(door_png)

    cfg = types.GenerateContentConfig(
        response_modalities=["Image"],
        image_config=types.ImageConfig(aspect_ratio=aspect),  # vertical 3:4
    )
    resp = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[prompt, door_img],
        config=cfg,
    )
    return as_image_bytes_from_response(resp)

# -------------------- BOT HANDLERS --------------------
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
    await m.answer("Выбери модель двери для вставки в интерьер:", reply_markup=build_doors_keyboard(0))
    await state.set_state(Flow.selecting_door)

@router.callback_query(F.data == "check_sub")
async def check_sub(cb: CallbackQuery, state: FSMContext):
    ok = await ensure_subscribed(cb.from_user.id)
    if not ok:
        await cb.answer("Ты ещё не подписан(а).", show_alert=True)
        return
    await cb.message.answer("Спасибо! Теперь выбери модель двери:", reply_markup=build_doors_keyboard(0))
    await state.set_state(Flow.selecting_door)
    await cb.answer()

@router.callback_query(Flow.selecting_door, F.data.startswith("page:"))
async def paginate(cb: CallbackQuery):
    page = int(cb.data.split(":")[1])
    await cb.message.edit_reply_markup(reply_markup=build_doors_keyboard(page))
    await cb.answer()

@router.callback_query(Flow.selecting_door, F.data.startswith("door:"))
async def chose_door(cb: CallbackQuery, state: FSMContext):
    door_id = cb.data.split(":")[1]
    door = next((d for d in CATALOG if str(d["id"]) == str(door_id)), None)
    if not door:
        await cb.answer("Не удалось найти эту дверь.", show_alert=True)
        return

    await state.update_data(door_id=door_id)
    await cb.message.answer(f"Модель выбрана: <b>{door['name']}</b>\nНачинаю генерацию…")
    await cb.answer()
    await state.set_state(Flow.generating)

    # --- GENERATE ---
    try:
        door_png = Path(door["image_png"])
        if not door_png.exists():
            await cb.message.answer(f"Файл двери не найден: {door_png}")
            await state.clear()
            return

        img_bytes = gemini_generate(door_png=door_png, aspect="3:4")
        try:
            file = BufferedInputFile(img_bytes, filename="result.png")
            await cb.message.answer_photo(photo=file, caption=f"{door['name']} — дверь по центру задней стены (белые панели)")
        except Exception:
            tmp = Path("/tmp") / f"{uuid.uuid4().hex}.png"
            tmp.write_bytes(img_bytes)
            await cb.message.answer_photo(photo=FSInputFile(str(tmp)), caption=f"{door['name']} — дверь по центру задней стены (белые панели)")
    except Exception as e:
        print("GENERATION_ERROR:", repr(e))
        await cb.message.answer("⚠️ Не удалось сгенерировать изображение. Проверьте ключ GEMINI_API_KEY и попробуйте ещё раз.")
    finally:
        await state.clear()

# -------------------- FASTAPI --------------------
app = FastAPI()

@app.get("/")
async def _health():
    return {"ok": True}

@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    update = await request.json()
    await dp.feed_webhook_update(bot, update)
    return {"ok": True}
