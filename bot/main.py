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

BASE_DIR = Path(__file__).resolve().parent.parent

# Список пользователей, которым отправляем картинки БЕЗ водяного знака
WATERMARK_WHITELIST_USERNAMES = {
    "Vlodekteper", 
}

# Если хочешь по ID (они стабильнее, чем username):
WATERMARK_WHITELIST_IDS: set[int] = set()



WATERMARK_PATH = BASE_DIR / "assets" / "watermark.png"
WATERMARK_ALPHA = 0.45        # прозрачность (0.0–1.0)
WATERMARK_WIDTH_RATIO = 0.25  # ширина водяного знака ~25% ширины картинки
WATERMARK_MARGIN_RATIO = 0.03 # отступ от краёв ~3% ширины

# =========================== ENV ==========================
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

STYLE_OPTIONS: List[Tuple[str, str, str]] = [
    ("scandi", "Скандинавский", "Scandinavian interior"),
    ("japandi", "Japandi", "Japandi interior"),
    ("minimal", "Современный минимализм", "Modern minimalist interior"),
    ("modern_classic", "Современная классика", "Modern classic interior"),
    ("loft", "Лофт / Индустриальный", "Industrial loft interior"),
    ("contemporary", "Контемпорари", "Contemporary interior"),
    ("midcentury", "Mid-century modern", "Mid-century modern interior"),
    ("wabi_sabi", "Ваби-саби", "Wabi-sabi interior"),
    ("farmhouse", "Фармхаус / Modern farmhouse", "Modern farmhouse interior"),
    ("transitional", "Переходный (Transitional)", "Transitional interior"),
]

# =========================== FSM ===========================
class Flow(StatesGroup):
    waiting_disclaimer_ok = State()
    choosing_mode = State()
    waiting_foto = State()
    waiting_text_palette = State()
    selecting_style = State()
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

# =========================== INTERIOR JSON ANALYSIS (Gemini 2.5 Flash) ===========================
INTERIOR_JSON_PROMPT = textwrap.dedent("""
You are a professional interior designer and door color specialist.

Your task is to analyze the interior and return a STRICTLY valid JSON containing three parts:

a short interior description + suitable door colors (with RAL codes),

a table of styles with probabilities,

and a list of recommended colors for interface buttons.
Write EVERYTHING in RUSSIAN.

Choose door colors only from real existing RAL colors (e.g., RAL 9016, RAL 9003, RAL 7047, RAL 7021, etc.).

Do NOT invent non-existent RAL codes.
JSON structure:


{
  "summary": {
    "interior_description": "short interior description, 2–4 sentences",
    "door_colors": [
      {
        "ral": "RAL 9016",
        "name": "Белый трафик",
        "why": "Short explanation of why this color matches the interior."
      }
    ]
  },
  "styles": {
    "Minimalism": 0,
    "Contemporary": 0,
    "Loft (Industrial)": 0,
    "Scandinavian": 0,
    "High-Tech": 0,
    "Eco Style": 0,
    "Mid-Century Modern": 0,
    "Japandi": 0,
    "Boho": 0,
    "Fusion": 0,
    "Eclectic": 0,
    "Maximalism": 0,
    "Wabi-Sabi": 0,
    "Hygge": 0,
    "Rustic (incl. Modern Rustic)": 0,
    "Farmhouse (Modern Country)": 0,
    "Grunge": 0,
    "Pop Art": 0,
    "Brutalism": 0,
    "Postmodernism": 0,
    "Memphis": 0,
    "Shabby Chic": 0,
    "Vintage": 0,
    "Retro": 0,
    "Bionic (Organic Tech)": 0,
    "Techno": 0,
    "Futurism": 0,
    "Steampunk": 0,
    "Kitsch": 0,
    "Lounge": 0,
    "Military": 0,
    "Bauhaus": 0,
    "Constructivism": 0,
    "Functionalism": 0,
    "De Stijl": 0
  },
  "recommended_colors": [
    {
      "ral": "RAL 9016",
      "name": "Белый трафик",
      "label": "RAL 9016 — Белый трафик"
    }
  ]
}
Rules:

summary.interior_description
Provide a short description of the interior: room type, atmosphere, materials, color palette.
summary.door_colors
List 2–5 best matching door colors.
For EACH:
"ral" — real RAL code
"name" — short Russian name of the color
"why" — one-sentence justification
styles
MUST include ALL of these styles (exact strings in English):
Minimalism
Contemporary
Loft (Industrial)
Scandinavian
High-Tech
Eco Style
Mid-Century Modern
Japandi
Boho
Fusion
Eclectic
Maximalism
Wabi-Sabi
Hygge
Rustic (incl. Modern Rustic)
Farmhouse (Modern Country)
Grunge
Pop Art
Brutalism
Postmodernism
Memphis
Shabby Chic
Vintage
Retro
Bionic (Organic Tech)
Techno
Futurism
Steampunk
Kitsch
Lounge
Military
Bauhaus
Constructivism
Functionalism
De Stijl
For each style output an integer probability (0–100).
They do not need to sum to 100.
recommended_colors
Provide 2–5 recommended color options for use as UI buttons.
For each:
"ral" — RAL code
"name" — short Russian color name
"label" — readable label like "RAL 9016 — Белый трафик"
Do NOT add any extra text, comments, or fields.
Return ONLY the JSON, with no explanations before or after.
""").strip()


def _parse_interior_json(txt: str) -> Optional[dict]:
    txt = txt.strip()
    if not txt:
        return None

    try:
        return json.loads(txt)
    except Exception as e:
        print("INTERIOR_JSON_PARSE_ERROR (direct):", repr(e))

    j = extract_json_block(txt)
    if not j:
        print("INTERIOR_JSON_EXTRACT_FAILED, raw snippet:", txt[:500])
    return j




async def analyze_scene_json_from_image(image_path: Path) -> Optional[dict]:
    """JSON-анализ интерьера по фото."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    img = Image.open(image_path).convert("RGB")

    cfg = types.GenerateContentConfig(
        temperature=0.2,
        response_mime_type="application/json",
    )

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[INTERIOR_JSON_PROMPT, img],
        config=cfg,
    )
    txt = _resp_text(resp).strip()
    try:
        return json.loads(txt)
    except Exception as e:
        print("INTERIOR_JSON_IMAGE_PARSE_ERROR:", repr(e))
        return _parse_interior_json(txt)




async def analyze_scene_json_from_text_and_palette(
    description_text: str,
    palette_image_path: Optional[Path],
) -> Optional[dict]:
    """JSON-анализ интерьера по тексту и, опционально, палитре."""
    client = genai.Client(api_key=GEMINI_API_KEY)

    contents: List[Any] = [INTERIOR_JSON_PROMPT, description_text.strip()]
    if palette_image_path is not None and palette_image_path.exists():
        img = Image.open(palette_image_path).convert("RGB")
        contents.append(img)

    cfg = types.GenerateContentConfig(
        temperature=0.2,
        response_mime_type="application/json",
    )

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=cfg,
    )
    txt = _resp_text(resp).strip()
    try:
        return json.loads(txt)
    except Exception as e:
        print("INTERIOR_JSON_TEXT_PALETTE_PARSE_ERROR:", repr(e))
        return _parse_interior_json(txt)



async def analyze_scene_json_from_style(style_prompt: str) -> Optional[dict]:
    """JSON-анализ интерьера по выбранному стилю (без фото)."""
    client = genai.Client(api_key=GEMINI_API_KEY)

    contents = [
        INTERIOR_JSON_PROMPT,
        f"Интерьер в стиле: {style_prompt}",
    ]

    cfg = types.GenerateContentConfig(
        temperature=0.2,
        response_mime_type="application/json",
    )

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=cfg,
    )
    txt = _resp_text(resp).strip()
    try:
        return json.loads(txt)
    except Exception as e:
        print("INTERIOR_JSON_STYLE_PARSE_ERROR:", repr(e))
        return _parse_interior_json(txt)




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
        on the description, hex range), interior objects with their shapes, sizes, and types,
        the lighting, the floor (type, texture, material, and color description and hex range), the walls (material and color description and hex range),
        the ceiling, and so on down to the smallest element. If the scene contains tiles, parquet, patterns on the wall, patterns on the floor, 
        their exact size must be indicated.
        In the description, you MUST NOT mention doors, doorways, or anything related to them.
        The description MUST NOT include the location of interior items, the shape of the room,
        or the location of anything in the interior at all.
        If the scene contains massive objects (tables, kitchen islands, sofas, beds) that serve as the center of the room, then when describing them, you 
        need to write that they are visible only at 20 or less percent of their volume, and do not write that they are the center of the room, don't 
        write that he's big.
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

async def describe_scene_from_text_and_palette(
    description_text: str,
    palette_image_path: Optional[Path],
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Описание интерьера по тексту пользователя и картинке-палитре.
    Возвращает:
      - english_description
      - recommended_colors (может быть пустым, если модель не вернёт JSON)
    """
    client = genai.Client(api_key=GEMINI_API_KEY)

    user_part = description_text.strip()
    base_prompt = textwrap.dedent(f"""
        You need to create a description of the interior {user_part} with the colors of this palette following the following interior design rules: 
        Describe this interior as thoroughly as possible. Style and type. Capture absolutely everything — every single detail —
        including all colors and the full color palette (Accuracy in the rendering of color and materials is very 
        important; the color must be described in such a way that any artist can easily draw identical materials based 
        on the description, hex range), interior objects with their shapes, sizes, and types,
        the lighting, the floor (type, texture, material, and color description and hex range), the walls (material and color description and hex range),
        the ceiling, and so on down to the smallest element. If the scene contains tiles, parquet, patterns on the wall, patterns on the floor, 
        their exact size must be indicated.
        In the description, you MUST NOT mention doors, doorways, or anything related to them.
        The description MUST NOT include the location of interior items, the shape of the room,
        or the location of anything in the interior at all.
        If the scene contains massive objects (tables, kitchen islands, sofas, beds) that serve as the center of the room, then when describing them, you 
        need to write that they are visible only at 20 or less percent of their volume, and do not write that they are the center of the room, don't 
        write that he's big.
        Write the description in English.
    """).strip()

    contents: List[Any] = [base_prompt]
    if palette_image_path is not None and palette_image_path.exists():
        img = Image.open(palette_image_path).convert("RGB")
        contents.append(img)

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(temperature=0.2),
    )

    txt = _resp_text(resp).strip()
    j = extract_json_block(txt)
    recommended = normalize_recommended_colors(j)

    english_description = txt
    if j:
        try:
            dumped = json.dumps(j, ensure_ascii=False)
            cut_pos = english_description.rfind(dumped)
            if cut_pos != -1:
                english_description = english_description[:cut_pos].strip()
        except Exception:
            pass

    return english_description, recommended


async def describe_scene_from_style(style_prompt: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Описание интерьера только по выбранному стилю (без исходного фото).
    """
    client = genai.Client(api_key=GEMINI_API_KEY)

    base_prompt = textwrap.dedent(f"""
        You need to create a description of the interior {style_prompt} following interior design rules: 
        Describe this interior as thoroughly as possible. Style and type. Capture absolutely everything — every single detail —
        including all colors and the full color palette (Accuracy in the rendering of color and materials is very 
        important; the color must be described in such a way that any artist can easily draw identical materials based 
        on the description, hex range), interior objects with their shapes, sizes, and types,
        the lighting, the floor (type, texture, material, and color description and hex range), the walls (material and color description and hex range),
        the ceiling, and so on down to the smallest element. If the scene contains tiles, parquet, patterns on the wall, patterns on the floor, 
        their exact size must be indicated.
        In the description, you MUST NOT mention doors, doorways, or anything related to them.
        The description MUST NOT include the location of interior items, the shape of the room,
        or the location of anything in the interior at all.
        If the scene contains massive objects (tables, kitchen islands, sofas, beds) that serve as the center of the room, then when describing them, you 
        need to write that they are visible only at 20 or less percent of their volume, and do not write that they are the center of the room, don't 
        write that he's big.
        Write the description in English.
    """).strip()

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[base_prompt],
        config=types.GenerateContentConfig(temperature=0.2),
    )

    txt = _resp_text(resp).strip()
    j = extract_json_block(txt)
    recommended = normalize_recommended_colors(j)

    english_description = txt
    if j:
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

def apply_watermark(image_bytes: bytes) -> bytes:
    """
    Накладывает большой полупрозрачный водяной знак, растянутый на всё изображение.
    Если watermark-файл не найден или ошибка — возвращаем исходные байты.
    """
    try:
        if not WATERMARK_PATH.exists():
            return image_bytes

        # исходное изображение
        base = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

        # watermark
        wm = Image.open(WATERMARK_PATH).convert("RGBA")

        # Масштабируем watermark так, чтобы он полностью перекрывал изображение
        # (cover: по большей стороне)
        scale = max(base.width / wm.width, base.height / wm.height)
        target_w = int(wm.width * scale)
        target_h = int(wm.height * scale)
        wm = wm.resize((target_w, target_h), Image.LANCZOS)

        # Ослабляем непрозрачность
        r, g, b, a = wm.split()
        a = a.point(lambda p: int(p * WATERMARK_ALPHA))
        wm = Image.merge("RGBA", (r, g, b, a))

        # Центруем watermark относительно кадра
        x = (base.width - wm.width) // 2
        y = (base.height - wm.height) // 2

        base.alpha_composite(wm, dest=(x, y))

        out = io.BytesIO()
        base.convert("RGB").save(out, format="PNG")
        return out.getvalue()
    except Exception as e:
        print("WATERMARK_ERROR:", repr(e))
        return image_bytes

from aiogram.types import User

def should_apply_watermark(user: Optional[User]) -> bool:
    """
    Возвращает True, если НУЖНО накладывать водяной знак.
    user передаём явно (а не через message.from_user),
    чтобы корректно работать и с callback'ами, и с message.
    """
    if user is None:
        return True  # безопасный дефолт — ставим водяной знак

    # Проверка по username
    if user.username and user.username.lower() in {u.lower() for u in WATERMARK_WHITELIST_USERNAMES}:
        return False

    # Если потом захочешь по ID:
    # if user.id in WATERMARK_WHITELIST_IDS:
    #     return False

    return True

STYLE_KEYS = [
    "Minimalism",
    "Contemporary",
    "Loft (Industrial)",
    "Scandinavian",
    "High-Tech",
    "Eco Style",
    "Mid-Century Modern",
    "Japandi",
    "Boho",
    "Fusion",
    "Eclectic",
    "Maximalism",
    "Wabi-Sabi",
    "Hygge",
    "Rustic (incl. Modern Rustic)",
    "Farmhouse (Modern Country)",
    "Grunge",
    "Pop Art",
    "Brutalism",
    "Postmodernism",
    "Memphis",
    "Shabby Chic",
    "Vintage",
    "Retro",
    "Bionic (Organic Tech)",
    "Techno",
    "Futurism",
    "Steampunk",
    "Kitsch",
    "Lounge",
    "Military",
    "Bauhaus",
    "Constructivism",
    "Functionalism",
    "De Stijl",
]


def compute_door_order(styles_profile: Dict[str, Any]) -> List[int]:
    """
    Считаем суммарную дельту |p_int - p_door| по всем стилям.
    Чем МЕНЬШЕ сумма, тем лучше дверь подходит.
    Возвращаем список ИНДЕКСОВ дверей в CATALOG в порядке возрастания этой суммы.
    """
    if not styles_profile:
        return list(range(len(CATALOG)))

    orders: List[Tuple[int, float]] = []

    for idx, door in enumerate(CATALOG):
        total_delta = 0.0
        for style in STYLE_KEYS:
            p_int = styles_profile.get(style, 0)
            try:
                p_int = float(p_int)
            except Exception:
                p_int = 0.0

            p_door = door.get(style, 0)
            try:
                p_door = float(p_door)
            except Exception:
                p_door = 0.0

            total_delta += abs(p_int - p_door)

        orders.append((idx, total_delta))

    orders.sort(key=lambda x: x[1])
    return [i for (i, _) in orders]

def extract_interior_profile(
    json_data: Optional[dict],
) -> Tuple[bool, str, Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Пытаемся вытащить данные из JSON второй модели.
    Возвращаем:
      - valid: bool — можно ли вообще использовать этот JSON?
      - summary_ru: краткое RU-описание
      - styles_profile: dict стилей
      - recommended_colors_json: список цветов для UI-кнопок
      - door_colors_json: список цветов двери с пояснением "why"
    Если что-то важное отсутствует — считаем JSON "битым" и возвращаем valid=False.
    """
    summary_ru = ""
    styles_profile: Dict[str, Any] = {}
    recommended_colors_json: List[Dict[str, Any]] = []
    door_colors_json: List[Dict[str, Any]] = []

    if not isinstance(json_data, dict):
        return False, summary_ru, styles_profile, recommended_colors_json, door_colors_json

    summary = json_data.get("summary") or {}
    styles = json_data.get("styles") or {}
    rec = json_data.get("recommended_colors") or []
    door_cols = summary.get("door_colors") or []

    summary_val = summary.get("interior_description")
    summary_ok = isinstance(summary_val, str) and summary_val.strip()
    styles_ok = isinstance(styles, dict) and len(styles) > 0
    rec_ok = isinstance(rec, list) and len(rec) > 0

    valid = bool(summary_ok and styles_ok and rec_ok)
    if not valid:
        print(
            "INTERIOR_JSON_VALIDATION_FAILED:",
            "summary_ok=", bool(summary_ok),
            "styles_ok=", bool(styles_ok),
            "rec_ok=", bool(rec_ok),
        )
        return False, "", {}, [], []

    summary_ru = summary_val.strip()
    styles_profile = styles
    recommended_colors_json = rec
    if isinstance(door_cols, list):
        door_colors_json = door_cols

    return True, summary_ru, styles_profile, recommended_colors_json, door_colors_json







# =========================== UI BUILDERS ===========================
BACK_INLINE_KB = InlineKeyboardMarkup(
    inline_keyboard=[[InlineKeyboardButton(text="⬅️ Назад", callback_data="back")]]
)

async def send_step_message(
    target,
    state: FSMContext,
    text: str,
    reply_markup: Optional[InlineKeyboardMarkup] = None,
    parse_mode: Optional[str] = "HTML",
):
    """
    Шаговое сообщение бота:
    - удаляет предыдущее служебное сообщение (если было);
    - отправляет новое и сохраняет его id в состоянии.
    ВАЖНО: НЕ использовать для итоговой фотографии.
    """
    # target может быть Message или CallbackQuery
    if isinstance(target, CallbackQuery):
        base_msg = target.message
    else:
        base_msg = target

    data = await state.get_data()
    last_id = data.get("last_bot_message_id")

    if last_id:
        try:
            await bot.delete_message(chat_id=base_msg.chat.id, message_id=last_id)
        except Exception:
            # Уже удалено / слишком старое — игнорируем
            pass

    sent = await base_msg.answer(text, parse_mode=parse_mode, reply_markup=reply_markup)
    await state.update_data(last_bot_message_id=sent.message_id)
    return sent


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
    rows.append([InlineKeyboardButton(text="⬅️ Назад", callback_data="back")])
    kb = InlineKeyboardMarkup(inline_keyboard=rows)
    description_text = "\n".join(description_lines) if description_lines else "Выберите один из предложенных оттенков или введите свой цвет."
    return kb, description_text

def build_styles_keyboard() -> InlineKeyboardMarkup:
    rows: List[List[InlineKeyboardButton]] = []
    row: List[InlineKeyboardButton] = []
    for style_id, label, _ in STYLE_OPTIONS:
        row.append(InlineKeyboardButton(text=label, callback_data=f"style:{style_id}"))
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton(text="⬅️ Назад", callback_data="back")])
    return InlineKeyboardMarkup(inline_keyboard=rows)



def current_catalog_index(state_data: Dict[str, Any]) -> int:
    return int(state_data.get("carousel_idx", 0))

def build_carousel_keyboard(idx: int) -> InlineKeyboardMarkup:
    nav = [
        InlineKeyboardButton(text="◀", callback_data="carousel:prev"),
        InlineKeyboardButton(text="✅ Выбрать", callback_data="carousel:choose"),
        InlineKeyboardButton(text="▶", callback_data="carousel:next"),
    ]
    back_row = [InlineKeyboardButton(text="⬅️ Назад", callback_data="back")]
    return InlineKeyboardMarkup(inline_keyboard=[nav, back_row])


def door_caption(door: Dict[str, Any], idx: int) -> str:
    total = len(CATALOG)
    return (
        f"<b>{door.get('name', 'Дверь')}</b>\n"
        f"Модель {idx+1} из {total}\n\n"
        "Выберите модель двери (листайте карусель):"
    )

async def show_or_update_carousel(cb_or_msg, state: FSMContext, idx: int):
    """
    Показываем/обновляем карусель с фото двери и кнопками.
    Используем порядок дверей из state['door_order'], если он есть.
    """
    data = await state.get_data()
    door_order: List[int] = data.get("door_order") or list(range(len(CATALOG)))

    if not door_order:
        door_order = list(range(len(CATALOG)))

    # idx – позиция в door_order
    idx = max(0, min(idx, len(door_order) - 1))
    await state.update_data(carousel_idx=idx)

    door_global_idx = door_order[idx]
    door = CATALOG[door_global_idx]

    img_path = Path(door["image_png"])
    caption = door_caption(door, idx)
    kb = build_carousel_keyboard(idx)

    # Определяем чат
    if isinstance(cb_or_msg, CallbackQuery):
        chat_id = cb_or_msg.message.chat.id
    else:
        chat_id = cb_or_msg.chat.id

    # Удаляем предыдущее шаговое сообщение (если было)
    last_id = data.get("last_bot_message_id")
    if last_id:
        try:
            await bot.delete_message(chat_id=chat_id, message_id=last_id)
        except Exception:
            pass

    sent_msg = None

    if isinstance(cb_or_msg, CallbackQuery):
        try:
            media = InputMediaPhoto(
                media=FSInputFile(str(img_path)),
                caption=caption,
                parse_mode="HTML",
            )
            await cb_or_msg.message.edit_media(media=media, reply_markup=kb)
            sent_msg = cb_or_msg.message  # id остаётся тем же
        except Exception:
            sent_msg = await cb_or_msg.message.answer_photo(
                photo=FSInputFile(str(img_path)),
                caption=caption,
                parse_mode="HTML",
                reply_markup=kb,
            )
    else:
        sent_msg = await cb_or_msg.answer_photo(
            photo=FSInputFile(str(img_path)),
            caption=caption,
            parse_mode="HTML",
            reply_markup=kb,
        )

    if sent_msg is not None:
        await state.update_data(last_bot_message_id=sent_msg.message_id)



# =========================== TELEGRAM BOT FLOW ===========================
async def send_disclaimer(msg: Message, state: FSMContext):
    disclaimer_text = (
        "⚠️ <b>Важный дисклеймер</b>\n\n"
        "Этот бот помогает получить общее представление ..."
    )
    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="OK", callback_data="disclaimer_ok")]
        ]
    )
    await state.set_state(Flow.waiting_disclaimer_ok)
    await send_step_message(msg, state, disclaimer_text, reply_markup=kb)


@router.message(CommandStart())
async def start(m: Message, state: FSMContext):
    ok = await ensure_subscribed(m.from_user.id)
    if not ok:
        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="Подписаться на канал", url=f"https://t.me/{REQUIRED_BUILDER2112.strip('@')}")],
            [InlineKeyboardButton(text="✅ Проверить подписку", callback_data="check_sub")],
        ])
        await send_step_message(
            m,
            state,
            "Чтобы пользоваться ботом, подпишись на наш канал и нажми «Проверить подписку».",
            reply_markup=kb,
            parse_mode=None,
        )
        return

    await send_disclaimer(m, state)


@router.callback_query(F.data == "check_sub")
async def check_sub(cb: CallbackQuery, state: FSMContext):
    ok = await ensure_subscribed(cb.from_user.id)
    if not ok:
        await cb.answer("Ты ещё не подписан(а).", show_alert=True)
        return

    await send_disclaimer(cb.message, state)
    await cb.answer("Подписка подтверждена!")

MODE_TEXT = (
    "Ваш интерьер может быть описан тремя способами:\n\n"
    "1. <b>Отправить фото интерьера / проекта</b> — мы проанализируем изображение и опишем интерьер без упоминания дверей.\n"
    "2. <b>Описать интерьер словами и приложить палитру</b> — вы пишете, что хотите видеть, и отправляете фото/скрин палитры цветов.\n"
    "3. <b>Выбрать стиль из списка</b> — мы создадим интерьер по популярному стилю, а потом вы выберете дверь и цвет.\n\n"
    "Выберите один из вариантов ниже:"
)

def build_mode_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📷 Отправить фото интерьера/проекта", callback_data="mode:photo")],
            [InlineKeyboardButton(text="📝 Текст + палитра", callback_data="mode:text_palette")],
            [InlineKeyboardButton(text="🎨 Выбрать стиль", callback_data="mode:style")],
        ]
    )


async def send_mode_menu(msg: Message, state: FSMContext):
    await send_step_message(
        msg,
        state,
        MODE_TEXT,
        reply_markup=build_mode_keyboard(),
        parse_mode="HTML",
    )
    await state.set_state(Flow.choosing_mode)



@router.callback_query(Flow.waiting_disclaimer_ok, F.data == "disclaimer_ok")
async def disclaimer_ok(cb: CallbackQuery, state: FSMContext):
    await send_mode_menu(cb.message, state)
    await cb.answer()

@router.callback_query(Flow.choosing_mode, F.data == "mode:photo")
async def mode_photo(cb: CallbackQuery, state: FSMContext):
    await state.update_data(entry_mode="photo")
    await send_step_message(
        cb,
        state,
        "Пришлите <b>фотографию интерьера</b> или дизайн-проекта. "
        "Мы опишем сцену и дальше предложим выбрать дверь.",
        reply_markup=BACK_INLINE_KB,
        parse_mode="HTML",
    )
    await state.set_state(Flow.waiting_foto)
    await cb.answer()




@router.callback_query(Flow.choosing_mode, F.data == "mode:text_palette")
async def mode_text_palette(cb: CallbackQuery, state: FSMContext):
    text = (
        "Опишите, пожалуйста, ваш интерьер словами и приложите <b>палитру</b> цветов:\n\n"
        "• Можно отправить одно сообщение с картинкой палитры и описанием в подписи.\n"
        "• Либо сначала текст, потом отдельным сообщением — скрин/фото палитры.\n\n"
        "Как только у нас будет и текст, и палитра, мы создадим детальное описание интерьера на их основе."
    )
    await send_step_message(cb, state, text, reply_markup=BACK_INLINE_KB, parse_mode="HTML")
    await state.update_data(tp_description=None, tp_palette_path=None, entry_mode="text_palette")
    await state.set_state(Flow.waiting_text_palette)
    await cb.answer()




@router.callback_query(Flow.choosing_mode, F.data == "mode:style")
async def mode_style(cb: CallbackQuery, state: FSMContext):
    await state.update_data(entry_mode="style")
    await send_step_message(
        cb,
        state,
        "Выберите интерьерный стиль, по которому мы создадим описание комнаты. "
        "Дальше вы сможете выбрать дверь и цвет.",
        reply_markup=build_styles_keyboard(),
        parse_mode=None,
    )
    await state.set_state(Flow.selecting_style)
    await cb.answer()


@router.callback_query(F.data == "back")
async def go_back(cb: CallbackQuery, state: FSMContext):
    cur_state = await state.get_state()
    data = await state.get_data()
    
    # Если по какой-то причине состояния нет — просто в начало
    if cur_state is None:
        await state.clear()
        await start(cb.message, state)
        await cb.answer()
        return

    # 1) Из меню выбора режима — назад к дисклеймеру
    if cur_state == Flow.choosing_mode.state:
        await send_disclaimer(cb.message, state)

    # 2) Ждали фото — назад к выбору режима
    elif cur_state == Flow.waiting_foto.state:
        await send_mode_menu(cb.message, state)

    # 3) Ждали текст+палитру — чистим временные данные и назад к выбору режима
    elif cur_state == Flow.waiting_text_palette.state:
        await state.update_data(tp_description=None, tp_palette_path=None)
        await send_mode_menu(cb.message, state)

    # 4) Выбор стиля — назад к выбору режима
    elif cur_state == Flow.selecting_style.state:
        await send_mode_menu(cb.message, state)

    # 5) Выбор двери — назад зависит от entry_mode
    elif cur_state == Flow.selecting_door.state:
        entry_mode = data.get("entry_mode")
        if entry_mode == "photo":
            await state.set_state(Flow.waiting_foto)
            await send_step_message(
                cb,
                state,
                "Пришлите <b>фотографию интерьера</b> или дизайн-проекта. "
                "Мы опишем сцену и дальше предложим выбрать дверь.",
                reply_markup=BACK_INLINE_KB,
                parse_mode="HTML",
            )
        elif entry_mode == "text_palette":
            await state.set_state(Flow.waiting_text_palette)
            text = (
                "Опишите, пожалуйста, ваш интерьер словами и приложите <b>палитру</b> цветов:\n\n"
                "• Можно отправить одно сообщение с картинкой палитры и описанием в подписи.\n"
                "• Либо сначала текст, потом отдельным сообщением — скрин/фото палитры.\n\n"
                "Как только у нас будет и текст, и палитра, мы создадим детальное описание интерьера на их основе."
            )
            await send_step_message(cb, state, text, reply_markup=BACK_INLINE_KB, parse_mode="HTML")
        elif entry_mode == "style":
            await state.set_state(Flow.selecting_style)
            await send_step_message(
                cb,
                state,
                "Выберите интерьерный стиль, по которому мы создадим описание комнаты. "
                "Дальше вы сможете выбрать дверь и цвет.",
                reply_markup=build_styles_keyboard(),
                parse_mode=None,
            )
        else:
            await send_mode_menu(cb.message, state)


    # 6) Выбор цвета — назад к карусели дверей
    elif cur_state == Flow.selecting_color.state:
        idx = current_catalog_index(data)
        await state.set_state(Flow.selecting_door)
        await show_or_update_carousel(cb.message, state, idx=idx)

    # 7) После результата — назад = «выбрать другую дверь»
    elif cur_state == Flow.after_result.state:
        await again_door(cb, state)
        # again_door сам поставит состояние и отправит нужные сообщения

    # 8) На всякий случай — в начало
    else:
        await state.clear()
        await start(cb.message, state)

    await cb.answer()


@router.message(Flow.waiting_text_palette)
async def handle_text_palette(m: Message, state: FSMContext):
    if not await ensure_subscribed(m.from_user.id):
        return

    data = await state.get_data()
    desc = (data.get("tp_description") or "").strip()
    palette_path = data.get("tp_palette_path")

    updated = False

    # Если прилетела картинка палитры
    if m.photo:
        workdir = Path("work") / str(m.from_user.id) / str(uuid.uuid4())
        img_path = workdir / "palette.jpg"
        await tg_download_photo(m, img_path)
        palette_path = str(img_path)
        updated = True
        # Берём описание из подписи, если оно есть
        if m.caption and m.caption.strip():
            desc = m.caption.strip()

    # Если прилетел только текст
    if m.text and m.text.strip():
        desc = m.text.strip()
        updated = True

    if not updated:
        await m.answer("Пожалуйста, отправьте текстовое описание интерьера и/или фото палитры.")
        return

    await state.update_data(tp_description=desc, tp_palette_path=palette_path)

    # Если у нас уже есть и описание, и палитра — запускаем генерацию описания
    if desc and palette_path:
        await run_text_palette_pipeline(m, state)
    elif desc and not palette_path:
        await m.answer("Отлично, описание получили. Теперь, пожалуйста, отправьте фото/скрин палитры.")
    elif palette_path and not desc:
        await m.answer("Палитру получили. Теперь, пожалуйста, отправьте текстовое описание интерьера.")

async def run_text_palette_pipeline(m: Message, state: FSMContext):
    data = await state.get_data()
    desc = (data.get("tp_description") or "").strip()
    palette_path_str = data.get("tp_palette_path")

    if not desc or not palette_path_str:
        await m.answer("Нужно и текстовое описание, и палитра, чтобы продолжить.")
        return

    palette_path = Path(palette_path_str)

    await state.set_state(Flow.describing)
    await send_step_message(
        m,
        state,
        "⏳ Пожалуйста, ожидайте: создаём интерьер по вашему описанию и палитре…",
    )

    typing_stop = asyncio.Event()
    typing_task = asyncio.create_task(
        run_chat_action(m.chat.id, ChatAction.TYPING, typing_stop)
    )

    english_desc = ""
    rec_colors_1: List[Dict[str, str]] = []
    json_data: Optional[dict] = None

    try:
        task_desc = asyncio.create_task(
            describe_scene_from_text_and_palette(desc, palette_path)
        )
        task_json = asyncio.create_task(
            analyze_scene_json_from_text_and_palette(desc, palette_path)
        )

        english_desc, rec_colors_1 = await task_desc
        json_data = await task_json

        if isinstance(json_data, dict):
            print(
                "DEBUG_INTERIOR_JSON_TEXT_PALETTE:",
                json.dumps(json_data, ensure_ascii=False)[:800],
            )
        else:
            print("DEBUG_INTERIOR_JSON_TEXT_PALETTE (non-dict):", json_data)
    finally:
        typing_stop.set()
        try:
            await typing_task
        except Exception:
            pass

    # --------- Разбор JSON второй модели с валидацией ---------
    interior_json_valid = False
    summary_ru = ""
    styles_profile: Dict[str, Any] = {}
    recommended_colors_json: List[Dict[str, Any]] = []
    door_colors_json: List[Dict[str, Any]] = []

    if json_data is not None:
        interior_json_valid, summary_ru, styles_profile, recommended_colors_json, door_colors_json = (
            extract_interior_profile(json_data)
        )

    # Если JSON валиден — показываем краткое RU-описание.
    # Если битый — НЕ показываем вообще ничего.
    if interior_json_valid:
        to_show = summary_ru.strip()
        if to_show:
            for chunk in textwrap.wrap(
                to_show, 3500, replace_whitespace=False, drop_whitespace=False
            ):
                await m.answer(truncate(chunk), parse_mode=None)

    # Порядок дверей: по стилям только если JSON валиден
    door_order = compute_door_order(styles_profile) if interior_json_valid else None

    payload: Dict[str, Any] = {
        "interior_description_en": english_desc,
        "styles_profile": styles_profile if interior_json_valid else {},
        "summary_ru": summary_ru if interior_json_valid else "",
        "recommended_colors_json": recommended_colors_json if interior_json_valid else [],
        "door_colors_json": door_colors_json if interior_json_valid else [],
        "interior_path": str(palette_path),
        "tp_description": None,
        "tp_palette_path": None,
        "interior_json_valid": interior_json_valid,
    }
    if door_order is not None:
        payload["door_order"] = door_order

    await state.update_data(**payload)

    await state.set_state(Flow.selecting_door)
    await show_or_update_carousel(m, state, idx=0)




@router.callback_query(Flow.selecting_style, F.data.startswith("style:"))
async def style_selected(cb: CallbackQuery, state: FSMContext):
    style_id = cb.data.split(":", 1)[1]
    style_entry = next((s for s in STYLE_OPTIONS if s[0] == style_id), None)
    if not style_entry:
        await cb.answer("Стиль не найден", show_alert=True)
        return

    await cb.answer()
    _, label_ru, style_prompt = style_entry

    await state.set_state(Flow.describing)
    await send_step_message(
        cb,
        state,
        f"⏳ Создаём интерьер в стиле «{label_ru}»…",
    )

    typing_stop = asyncio.Event()
    typing_task = asyncio.create_task(
        run_chat_action(cb.message.chat.id, ChatAction.TYPING, typing_stop)
    )

    english_desc = ""
    rec_colors_1: List[Dict[str, str]] = []
    json_data: Optional[dict] = None

    try:
        task_desc = asyncio.create_task(
            describe_scene_from_style(style_prompt)
        )
        task_json = asyncio.create_task(
            analyze_scene_json_from_style(style_prompt)
        )

        english_desc, rec_colors_1 = await task_desc
        json_data = await task_json

        if isinstance(json_data, dict):
            print(
                "DEBUG_INTERIOR_JSON_STYLE:",
                json.dumps(json_data, ensure_ascii=False)[:800],
            )
        else:
            print("DEBUG_INTERIOR_JSON_STYLE (non-dict):", json_data)
    finally:
        typing_stop.set()
        try:
            await typing_task
        except Exception:
            pass

    interior_json_valid = False
    summary_ru = ""
    styles_profile: Dict[str, Any] = {}
    recommended_colors_json: List[Dict[str, Any]] = []
    door_colors_json: List[Dict[str, Any]] = []

    if json_data is not None:
        interior_json_valid, summary_ru, styles_profile, recommended_colors_json, door_colors_json = (
            extract_interior_profile(json_data)
        )

    # Показываем краткое RU-описание ТОЛЬКО если JSON валиден.
    if interior_json_valid:
        to_show = summary_ru.strip()
        if to_show:
            for chunk in textwrap.wrap(
                to_show, 3500, replace_whitespace=False, drop_whitespace=False
            ):
                await cb.message.answer(truncate(chunk), parse_mode=None)

    door_order = compute_door_order(styles_profile) if interior_json_valid else None

    payload: Dict[str, Any] = {
        "interior_description_en": english_desc,
        "styles_profile": styles_profile if interior_json_valid else {},
        "summary_ru": summary_ru if interior_json_valid else "",
        "recommended_colors_json": recommended_colors_json if interior_json_valid else [],
        "door_colors_json": door_colors_json if interior_json_valid else [],
        "interior_json_valid": interior_json_valid,
    }
    if door_order is not None:
        payload["door_order"] = door_order

    await state.update_data(**payload)

    await state.set_state(Flow.selecting_door)
    await show_or_update_carousel(cb.message, state, idx=0)

    # ЗДЕСЬ больше НЕ нужно cb.answer()


@router.message(Flow.waiting_foto, F.photo)
async def got_photo(m: Message, state: FSMContext):
    if not await ensure_subscribed(m.from_user.id):
        return

    workdir = Path("work") / str(m.from_user.id) / str(uuid.uuid4())
    img_path = workdir / "interior.jpg"
    await tg_download_photo(m, img_path)

    await state.set_state(Flow.describing)
    await send_step_message(
        m,
        state,
        "⏳ Пожалуйста, ожидайте: происходит загрузка и анализ вашего изображения…",
    )

    typing_stop = asyncio.Event()
    typing_task = asyncio.create_task(
        run_chat_action(m.chat.id, ChatAction.TYPING, typing_stop)
    )

    english_desc = ""
    rec_colors_1: List[Dict[str, str]] = []
    json_data: Optional[dict] = None

    try:
        # ДВА запроса параллельно:
        # 1) старое детальное англ. описание (для генерации изображения)
        # 2) новый JSON с кратким описанием, стилями и цветами
        task_desc = asyncio.create_task(describe_scene_with_gemini(img_path))
        task_json = asyncio.create_task(analyze_scene_json_from_image(img_path))

        english_desc, rec_colors_1 = await task_desc
        json_data = await task_json

        # лог для дебага — можно потом убрать
        if isinstance(json_data, dict):
            print(
                "DEBUG_INTERIOR_JSON:",
                json.dumps(json_data, ensure_ascii=False)[:800],
            )
        else:
            print("DEBUG_INTERIOR_JSON (non-dict):", json_data)
    finally:
        typing_stop.set()
        try:
            await typing_task
        except Exception:
            pass

    # --------- Разбор JSON второй модели с валидацией ---------
    interior_json_valid = False
    summary_ru = ""
    styles_profile: Dict[str, Any] = {}
    recommended_colors_json: List[Dict[str, Any]] = []
    door_colors_json: List[Dict[str, Any]] = []

    if json_data is not None:
        interior_json_valid, summary_ru, styles_profile, recommended_colors_json, door_colors_json = (
            extract_interior_profile(json_data)
        )

    # 1) Что показываем пользователю:
    #    - если JSON валиден → краткое RU-описание
    #    - если JSON битый → НИЧЕГО не показываем (даже english_desc)
    if interior_json_valid:
        to_show = summary_ru.strip()
        if to_show:
            for chunk in textwrap.wrap(
                to_show, 3500, replace_whitespace=False, drop_whitespace=False
            ):
                await m.answer(truncate(chunk), parse_mode=None)

    # 2) Порядок дверей:
    #    - если JSON валиден → сортируем по стилям
    #    - если нет → не записываем door_order, карусель покажет CATALOG как есть
    door_order = compute_door_order(styles_profile) if interior_json_valid else None

    # 3) Сохраняем всё нужное в state
    payload: Dict[str, Any] = {
        "interior_path": str(img_path),
        "interior_description_en": english_desc,  # для генерации картинки
        "styles_profile": styles_profile if interior_json_valid else {},
        "summary_ru": summary_ru if interior_json_valid else "",
        "recommended_colors_json": recommended_colors_json if interior_json_valid else [],
        "door_colors_json": door_colors_json if interior_json_valid else [],
        "carousel_idx": 0,
        "interior_json_valid": interior_json_valid,
    }
    if door_order is not None:
        payload["door_order"] = door_order

    await state.update_data(**payload)

    # 4) Карусель дверей:
    #    - если есть door_order → по стилям
    #    - если нет → в дефолтном порядке CATALOG
    await state.set_state(Flow.selecting_door)
    await show_or_update_carousel(m, state, idx=0)




@router.callback_query(Flow.selecting_door, F.data.startswith("carousel:"))
async def carousel_nav(cb: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    door_order: List[int] = data.get("door_order") or list(range(len(CATALOG)))

    if not door_order:
        door_order = list(range(len(CATALOG)))

    idx = current_catalog_index(data)
    action = cb.data.split(":")[1]

    if action == "prev":
        idx = (idx - 1) % len(door_order)
        await show_or_update_carousel(cb, state, idx)
        await cb.answer()
        return

    elif action == "next":
        idx = (idx + 1) % len(door_order)
        await show_or_update_carousel(cb, state, idx)
        await cb.answer()
        return

    elif action == "choose":
        # текущая дверь по стилевому порядку
        door_global_idx = door_order[idx]
        door = CATALOG[door_global_idx]
        await state.update_data(door_id=str(door["id"]))

        data = await state.get_data()
        json_ok = data.get("interior_json_valid", False)

        if not json_ok:
            # --- Fallback: JSON битый/невалидный ---
            # Используем только дефолтные цвета из каталога, без сообщения "Рекомендуемые цвета двери..."
            colors_catalog = door.get("colors") or []
            merged: List[Dict[str, str]] = []

            if colors_catalog:
                merged.extend(colors_catalog)

            if not merged:
                defaults = door.get("default_colors", []) or ["#FFFFFF", "#F3F0E6", "#1E1E1E"]
                for hx in defaults:
                    merged.append({"ral": hx, "name": hx, "reason_ru": ""})

            kb, _descr = build_colors_keyboard_and_text(merged)
            await state.update_data(available_colors=merged)

            await send_step_message(
                cb,
                state,
                f"Модель: <b>{door['name']}</b>\n\n"
                "Выберите цвет полотна и рамки (фурнитура НЕ перекрашивается):",
                reply_markup=kb,
                parse_mode="HTML",
            )
            await state.set_state(Flow.selecting_color)
            await cb.answer()
            return

        # --- JSON валиден: используем рекомендованные цвета и описания ---
        rec_colors = data.get("recommended_colors_json") or []
        door_colors_info = data.get("door_colors_json") or []

        enriched: List[Dict[str, str]] = []
        reasons_by_ral: Dict[str, str] = {}
        for d in door_colors_info:
            ral = str(d.get("ral", "")).strip().upper()
            if ral:
                reasons_by_ral[ral] = d.get("why", "").strip()

        for rc in rec_colors[:5]:
            ral = str(rc.get("ral", "")).strip()
            label = (rc.get("label") or "").strip()
            name = (rc.get("name") or "").strip()

            ral_norm = ral.upper()
            reason = reasons_by_ral.get(ral_norm, "")
            display_name = label or (f"{ral} — {name}" if ral and name else (ral or name or "Цвет"))

            enriched.append(
                {
                    "ral": ral,
                    "name": display_name,
                    "reason_ru": reason,
                }
            )

        # если по какой-то причине список пустой даже при json_ok — fallback на дефолтные
        if not enriched:
            defaults = door.get("default_colors", []) or ["#FFFFFF", "#F3F0E6", "#1E1E1E"]
            for hx in defaults:
                enriched.append({"ral": hx, "name": hx, "reason_ru": ""})

        kb, descr = build_colors_keyboard_and_text(enriched)
        await state.update_data(available_colors=enriched)

        # Отдельное сообщение с описанием рекомендованных цветов
        if descr:
            await cb.message.answer(
                "Рекомендуемые цвета двери для этого интерьера:\n\n" + descr
            )

        await send_step_message(
            cb,
            state,
            f"Модель: <b>{door['name']}</b>\n\n"
            "Выберите цвет полотна и рамки (фурнитура НЕ перекрашивается):",
            reply_markup=kb,
            parse_mode="HTML",
        )
        await state.set_state(Flow.selecting_color)
        await cb.answer()

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
        await generate_and_send(cb.message, state, cb.from_user)
    else:
        await cb.answer("Неверный выбор цвета", show_alert=True)

@router.callback_query(Flow.selecting_color, F.data == "color:custom")
async def ask_custom_color(cb: CallbackQuery, state: FSMContext):
    await send_step_message(
        cb,
        state,
        "Напишите цвет: #HEX (например <code>#F3F0E6</code>), или <code>RAL 9010</code>, или простым словом (white, beige…).",
        parse_mode="HTML",
    )
    await cb.answer()

@router.message(Flow.selecting_color)
async def typed_color(m: Message, state: FSMContext):
    color_user = parse_color(m.text or "")
    # Сохраним как есть для промпта, чтобы не терять RAL/имя
    if not color_user:
        await m.answer("Не удалось распознать цвет. Попробуйте снова: #HEX, RAL XXXX или название.")
        return
    await state.update_data(color_raw={"input": m.text.strip()}, color_text=color_user)
    await generate_and_send(m, state, m.from_user)

async def generate_and_send(m: Message, state: FSMContext, user: User):
    if not await ensure_subscribed(user.id):
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

    await send_step_message(
        m,
        state,
        "⏳ Пожалуйста, ожидайте: выполняется генерация вашего интерьера…\n\n"
        "<b>Важно!</b> Иногда изображения могут не соответствовать ожиданиям. "
        "Попробуйте выбрать другую модель двери или другой интерьер. "
        "Цвета на изображении носят ориентировочный характер.",
        parse_mode="HTML",
    )
    typing_stop = asyncio.Event()
    typing_task = asyncio.create_task(
        run_chat_action(m.chat.id, ChatAction.UPLOAD_PHOTO, typing_stop)
    )

    try:
        img_bytes = await gemini_generate(
            door_png=door_png,
            color_text=color_text,
            interior_en=interior_en,
            aspect="3:4",
        )

        # ВАЖНО: теперь сюда передаём именно user
        if should_apply_watermark(user):
            img_bytes = apply_watermark(img_bytes)

        try:
            file = BufferedInputFile(img_bytes, filename="result.png")
            await m.answer_photo(
                photo=file,
                caption=(
                    f"{door['name']} — выбранный цвет: {color_text}\n"
                    f"Дверь по центру задней стены, полностью видима (ничем не закрыта)."
                ),
            )
        except Exception:
            tmp = Path("/tmp") / f"{uuid.uuid4().hex}.png"
            tmp.write_bytes(img_bytes)
            await m.answer_photo(
                photo=FSInputFile(str(tmp)),
                caption=f"{door['name']} — выбранный цвет: {color_text}",
            )
    except Exception as e:
        print("GENERATION_ERROR:", repr(e))
        await m.answer(
            "⚠️ Не удалось сгенерировать изображение. Проверьте ключи и попробуйте ещё раз."
        )
    finally:
        typing_stop.set()
        try:
            await typing_task
        except Exception:
            pass

    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🔁 Выбрать другую дверь для этого интерьера",
                    callback_data="again:door",
                )
            ],
            [
                InlineKeyboardButton(
                    text="🆕 Начать заново с новым интерьером", callback_data="again:new"
                )
            ],
        ]
    )
    await send_step_message(m, state, "Что дальше?", reply_markup=kb)
    await state.set_state(Flow.after_result)




@router.callback_query(Flow.after_result, F.data == "again:door")
async def again_door(cb: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    if not data.get("interior_description_en"):
        await cb.message.answer("Сессия описания интерьера не найдена. Запустите заново: /start")
        await state.clear()
        await cb.answer()
        return

    # этот текст можно вообще не считать отдельным сообщением,
    # а просто сразу показать карусель:
    await state.set_state(Flow.selecting_door)
    await show_or_update_carousel(cb.message, state, idx=0)
    await cb.answer()

@router.callback_query(Flow.after_result, F.data == "again:new")
async def again_new(cb: CallbackQuery, state: FSMContext):
    await send_step_message(
        cb,
        state,
        "Пришлите новое фото интерьера.",
        reply_markup=BACK_INLINE_KB,
        parse_mode="HTML",
    )
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
