"""
Vietnamese Medical NER Pipeline (v3 - + Assertion Classification)
---------------------------------------------------------------------------------
Kien truc 2 tang cho NER (sub-problem 1), giai quyet van de LLM "bo cuoc som"
khi phai trich xuat nguyen mot van ban dai co nhieu muc liet ke:

  TANG 1 (rule-based, KHONG dung LLM):
      generate_candidates() - sinh danh sach candidate span bang cach cat
      van ban theo cac boundary co the doan truoc (so thu tu, tu khoa
      "dieu tri", dau cau...). Uu tien KHONG BO SOT (candidate co the gom
      thua tu ngu canh, se duoc LLM tinh chinh o tang 2).

  TANG 2 (LLM, pham vi nho - moi lan chi xu ly 1 candidate ngan):
      Voi moi candidate, LLM chi lam 2 viec nhe:
        (a) tinh chinh lai dung ranh gioi entity (bo tu ngu canh thua nhu
            "duoc chan doan mac benh", "co tien su su dung"...) - co the
            tach 1 candidate thanh NHIEU entity (vd "lo âu mất ngủ" -> 2
            trieu chung rieng biet)
        (b) phan loai type, hoac bo qua neu candidate khong chua khai
            niem y te nao (vd cau mo dau "Danh sach thuoc...")
      Vi pham vi rat nho (1 cum tu ngan), LLM 7B xu ly de dang va dang tin
      cay hon nhieu so voi bat no trich xuat toan bo van ban dai 1 luc.

Sau khi co refined text, vi tri tuyet doi trong van ban goc duoc tinh lai
bang string search TRUC TIEP TREN TOAN VAN BAN GOC (candidate_start chi
dung lam DIEM NEO de chon dung lan xuat hien gan nhat khi entity lap lai
nhieu cho). Co normalize Unicode NFC de tranh loi troi vi tri do tieng Viet
co dau bi lech chuan hoa giua van ban goc va output LLM.

ASSERTION CLASSIFICATION (sub-problem 3) - kien truc 3 tang:

  TANG A (rule-based, quet 1 lan toan van ban):
      find_scope_events() - tim cac diem "doi scope" dua tren cue phrase
      CU THE truoc, CHUNG CHUNG sau (tranh bay "Tiền sử bệnh hiện tại" -
      thuat ngu HPI, KHONG phai isHistorical du chua chu "tien su").
      Scope tim duoc ap dung cho MOI ENTITY sau do, den khi gap event moi
      (giai quyet dung case danh sach thuoc duoi header "truoc nhap vien"
      ma khong co tu "tien su" nao gan tung thuoc ca).

  TANG B (rule-based, NegEx-style, pham vi 1 menh de):
      find_local_triggers() - quet trigger CUC BO (isNegated/isFamily)
      trong menh de chua entity, dung WORD-BOUNDARY regex (tranh bay
      "không" chua san "ông" nhu substring).

  TANG C (LLM, pham vi nho - xac nhan/dieu chinh):
      classify_assertions() - LLM nhan candidate tu Tang A+B lam GOI Y,
      xac nhan hoac dieu chinh dua tren ngu canh cau that su.

Entity linking (ICD/RxNorm candidates - sub-problem 2) CHUA lam trong file
nay - se bo sung sau.
"""

import json
import re
import unicodedata
import difflib
from typing import List, Dict, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def normalize_text(text: str) -> str:
    """
    Chuan hoa Unicode ve dang NFC. QUAN TRONG: tieng Viet co dau co the bieu
    dien o 2 dang khac nhau o tang byte (NFC gon, NFD tach dau rieng) du
    hien thi giong het nhau. Neu van ban goc va output LLM lech chuan hoa,
    moi vi tri tinh tu do se bi TROI DAN (loi "lech chut it" ma khong sai
    hoan toan). Luon normalize ve NFC O MOI DIEM SO SANH/GHI DE de tranh loi nay.
    """
    return unicodedata.normalize("NFC", text)


# =============================================================================
# 1. LOAD MODEL
# =============================================================================

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

print(f"Loading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
print("Model loaded.")


ENTITY_TYPES = [
    "TRIỆU_CHỨNG",
    "TÊN_XÉT_NGHIỆM",
    "KẾT_QUẢ_XÉT_NGHIỆM",
    "CHẨN_ĐOÁN",
    "THUỐC",
]
NOT_ENTITY_LABEL = "KHÔNG_PHẢI"


# =============================================================================
# 2. TANG 1: RULE-BASED CANDIDATE GENERATION (khong dung LLM)
# =============================================================================

_BOUNDARY_PATTERN = re.compile(
    r'(?P<marker>(?:(?<=\s)|^)\d{1,2}\.\s+)'   # so thu tu "1. " "10. "
    r'|(?P<dieutri>\s*điều trị\s+)'             # tu khoa "dieu tri" (phan cach thuoc/chi dinh)
    r'|(?P<comma>,\s*(?!\d))'                   # dau phay KHONG nam giua 2 chu so
    r'|(?P<period>\.(?!\d)\s*)'                 # dau cham KHONG theo sau boi chu so
    r'|(?P<semicolon>;\s*)'                     # dau cham phay
    r'|(?P<colon>:\s*(?=\d))'                   # hai cham CHI KHI theo sau la chu so
)


def generate_candidates(text: str) -> List[Tuple[str, int, int, bool]]:
    """
    Sinh candidate spans bang rule-based segmentation.
    Tra ve list (candidate_text, start, end, is_indication).

    is_indication=True danh dau candidate la VE PHAI cua tu khoa "điều trị"
    (vd "táo bón" trong "...po bid điều trị táo bón") - day la CHI DINH/LY DO
    ke don thuoc, KHONG phai phat bieu doc lap ve trieu chung/tien su cua
    benh nhan. Cac candidate nay se BI LOAI TRU khoi assertion scope o buoc
    sau (xem extract_entities) - tranh loi gan nham isHistorical/isNegated
    cho van ban chi mo ta muc dich ke don, khong phai tinh trang benh nhan.
    """
    boundaries = list(_BOUNDARY_PATTERN.finditer(text))

    fragments = []  # (start, end, is_indication)
    pos = 0
    next_is_indication = False
    for b in boundaries:
        if b.start() > pos:
            fragments.append((pos, b.start(), next_is_indication))
        next_is_indication = (b.lastgroup == "dieutri")
        pos = b.end()
    if pos < len(text):
        fragments.append((pos, len(text), next_is_indication))

    candidates = []
    for start, end, is_indication in fragments:
        raw = text[start:end]
        stripped = raw.strip()
        if not stripped:
            continue
        lstrip_len = len(raw) - len(raw.lstrip())
        real_start = start + lstrip_len
        real_end = real_start + len(stripped)
        candidates.append((stripped, real_start, real_end, is_indication))

    return candidates


# =============================================================================
# 3. TANG 2: LLM REFINE + CLASSIFY (pham vi nho - moi candidate 1 cau hoi)
# =============================================================================

CLASSIFY_SYSTEM_PROMPT = f"""
You are a Vietnamese clinical text classifier. You will be given a short
candidate phrase extracted from a medical note. The phrase MAY contain:
  - extra surrounding words that are not part of any medical entity
    (e.g. connector phrases like "được chẩn đoán mắc bệnh", "có tiền sử sử dụng",
    "đã tiến hành ... bằng ...")
  - MORE THAN ONE distinct medical entity with NO separator between them
    (e.g. "lo âu mất ngủ" contains TWO separate symptoms: "lo âu" and "mất ngủ")

Your task:
1. Identify EVERY core medical entity span WITHIN the given phrase (there
   may be zero, one, or multiple entities in a single phrase).
   Each entity text must be a contiguous substring of the input phrase,
   character-for-character - do NOT paraphrase, do NOT add/remove characters.
   Strip away any connector/descriptive words that are not part of the
   entity itself.

   IMPORTANT rule for deciding whether to SPLIT a multi-word phrase with no
   separator into multiple entities (this rule ONLY applies when the input
   phrase contains TWO OR MORE words that could potentially be split - it
   does NOT mean a short/single-syllable word can never be an entity by
   itself; a standalone single-syllable word like "ho" (cough) IS still a
   complete, valid entity on its own when that is the entire input phrase):
   - SPLIT only when EACH resulting part is an independently-recognized,
     multi-syllable (2+ syllable) medical term that stands on its own
     (e.g. "lo âu" = anxiety, "mất ngủ" = insomnia - both are complete,
     well-known standalone symptom names).
   - Do NOT split if any resulting part would be a short, generic,
     single-syllable word (like "đau" = pain, "sốt" = fever) that commonly
     appears combined with an adjacent word as ONE compound/idiomatic
     phrase (e.g. "sốt đau" = "fever and pain", a standard combined
     indication for antipyretic/analgesic drugs - keep this as ONE entity,
     do NOT split into "sốt" + "đau").
   - When uncertain whether to split, prefer keeping the phrase as ONE
     entity rather than over-splitting.
   - This rule is ONLY about splitting compound phrases. It never means
     "reject" or "ignore" a short single-word phrase - if the ENTIRE input
     phrase is just one short word like "ho", it is still a valid entity.

2. Classify each entity into exactly one of: {", ".join(ENTITY_TYPES)}
3. If the phrase does not contain any medical entity at all (e.g. it is
   just an introductory sentence, a header, or filler text), return an
   empty array [].

Return ONLY a valid JSON ARRAY of objects, each with exactly two fields
"text" and "type". No explanation, no markdown code fences.

Examples

Input phrase: "được chẩn đoán mắc bệnh trào ngược dạ dày - thực quản"
Output: [{{"text": "trào ngược dạ dày - thực quản", "type": "CHẨN_ĐOÁN"}}]

Input phrase: "Bệnh nhân có tiền sử sử dụng Chlorpheniramine 0.4 MG/ML"
Output: [{{"text": "Chlorpheniramine 0.4 MG/ML", "type": "THUỐC"}}]

Input phrase: "đã tiến hành tổng phân tích tế bào máu bằng máy lazer (tbm): WBC"
Output: [{{"text": "WBC", "type": "TÊN_XÉT_NGHIỆM"}}]

Input phrase: "amlodipine 10 mg po daily"
Output: [{{"text": "amlodipine 10 mg po daily", "type": "THUỐC"}}]

Input phrase: "ho"
Output: [{{"text": "ho", "type": "TRIỆU_CHỨNG"}}]

Input phrase: "14,43"
Output: [{{"text": "14,43", "type": "KẾT_QUẢ_XÉT_NGHIỆM"}}]

Input phrase: "lo âu mất ngủ"
Output: [{{"text": "lo âu", "type": "TRIỆU_CHỨNG"}}, {{"text": "mất ngủ", "type": "TRIỆU_CHỨNG"}}]

Input phrase: "sốt đau"
Output: [{{"text": "sốt đau", "type": "TRIỆU_CHỨNG"}}]

Input phrase: "Danh sách thuốc trước nhập viện chính xác và đầy đủ"
Output: []
""".strip()


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def classify_candidate(candidate_text: str) -> List[Dict]:
    """
    Goi LLM voi 1 candidate NGAN. Tra ve list [{"text":..., "type":...}, ...]
    - CO THE co 0, 1, hoac NHIEU entity (vd candidate "lo âu mất ngủ" tra ve
    2 entity rieng biet). List rong neu candidate khong chua entity nao.
    """
    messages = [
        {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
        {"role": "user", "content": f'Input phrase: "{candidate_text}"'},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=192,   # candidate ngan nhung co the ra 2-3 entity -> tang nhe budget
        do_sample=False,
        temperature=None,
        top_p=None,
        eos_token_id=tokenizer.eos_token_id,
    )

    raw = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[-1]:],
        skip_special_tokens=True,
    )
    cleaned = _strip_code_fences(raw)

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
            except json.JSONDecodeError:
                print(f"[WARN] Khong parse duoc JSON cho candidate {candidate_text!r}:\n{raw}")
                return []
        else:
            print(f"[WARN] Khong tim thay JSON array cho candidate {candidate_text!r}:\n{raw}")
            return []

    if not isinstance(result, list):
        print(f"[WARN] LLM tra ve khong phai list cho candidate {candidate_text!r}: {result}")
        return []

    valid_results = []
    for item in result:
        if not isinstance(item, dict) or "text" not in item or "type" not in item:
            print(f"[WARN] Entity thieu field can thiet trong candidate {candidate_text!r}: {item}")
            continue
        if not item.get("text"):
            continue
        if item["type"] not in ENTITY_TYPES:
            print(f"[WARN] Type khong hop le '{item['type']}' cho candidate {candidate_text!r}")
            continue
        valid_results.append(item)

    return valid_results


# =============================================================================
# 4. DINH VI LAI: SO KHOP TRUC TIEP VOI TOAN VAN BAN GOC
# =============================================================================
#
# QUAN TRONG: refined_text (LLM tra ve) duoc so khop TRUC TIEP voi TOAN BO
# van ban goc (khong gioi han trong pham vi candidate). candidate_start CHI
# dung lam DIEM NEO de chon dung lan xuat hien gan nhat khi entity nay lap
# lai nhieu cho trong van ban (vd "táo bón" xuat hien 2 lan o 2 vi tri khac
# nhau) - neu khong co neo, .find() se luon chon lan xuat hien DAU TIEN,
# co the sai vi tri.
#
# 3 tang fallback theo do "chac chan" giam dan:
#   1. Tim chinh xac (sau khi normalize NFC ca 2 ben - xem normalize_text())
#   2. Normalize khoang trang (LLM sinh thua/thieu 1 space)
#   3. Fuzzy match bang difflib, GIOI HAN trong 1 cua so +-200 ky tu quanh
#      diem neo (LLM tu sua nhe chinh ta/viet hoa) - gioi han cua so de
#      tranh match nham sang cho khac xa trong van ban dai


def _find_all_occurrences(haystack: str, needle: str) -> List[int]:
    """Tra ve list vi tri start cua TAT CA lan xuat hien cua needle trong haystack."""
    if not needle:
        return []
    positions = []
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + 1
    return positions


def locate_refined_span(
    original_text: str, candidate_start: int, refined_text: str
) -> Tuple[Optional[int], Optional[int]]:
    """
    So khop refined_text (LLM da tinh chinh) TRUC TIEP voi TOAN BO original_text.
    candidate_start chi la diem neo de chon dung lan xuat hien gan nhat.
    Tra ve vi tri TUYET DOI (start, end) trong original_text, hoac (None, None)
    neu khong tim thay o ca 3 tang fallback.
    """
    text_norm = normalize_text(original_text)
    refined_norm = normalize_text(refined_text)

    def pick_closest(positions: List[int], length: int):
        if not positions:
            return None, None
        best = min(positions, key=lambda p: abs(p - candidate_start))
        return best, best + length

    # --- Tang 1: tim chinh xac, chon lan xuat hien GAN candidate_start nhat ---
    positions = _find_all_occurrences(text_norm, refined_norm)
    start, end = pick_closest(positions, len(refined_norm))
    if start is not None:
        return start, end

    # --- Tang 2: normalize khoang trang ---
    pattern = re.escape(refined_norm)
    pattern = re.sub(r"(\\ )+", r"\\s+", pattern)
    matches = list(re.finditer(pattern, text_norm))
    if matches:
        best = min(matches, key=lambda m: abs(m.start() - candidate_start))
        return best.start(), best.end()

    # --- Tang 3: fuzzy match bang difflib, gioi han trong cua so quanh diem neo ---
    window_start = max(0, candidate_start - 200)
    window_end = min(len(text_norm), candidate_start + len(refined_norm) + 200)
    window_text = text_norm[window_start:window_end]

    sm = difflib.SequenceMatcher(None, window_text, refined_norm, autojunk=False)
    match = sm.find_longest_match(0, len(window_text), 0, len(refined_norm))
    if match.size >= max(3, len(refined_norm) // 2):  # doi hoi it nhat ~50% khop
        local_start = max(0, match.a - match.b)
        local_end = min(len(window_text), local_start + len(refined_norm))
        if local_end > local_start:
            return window_start + local_start, window_start + local_end

    return None, None


# =============================================================================
# 5. ASSERTION CLASSIFICATION (sub-problem 3) - kien truc 3 tang
# =============================================================================

ASSERTION_LABELS = ["isNegated", "isFamily", "isHistorical"]
# Cac type duoc phep co assertions theo de bai (KET_QUA/TEN_XET_NGHIEM khong co)
ASSERTABLE_TYPES = {"TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC"}


# --- TANG A: SECTION/SCOPE-LEVEL CUES -------------------------------------
# Thu tu QUAN TRONG: cue CU THE/DAI kiem tra TRUOC, cue CHUNG CHUNG/NGAN
# kiem tra SAU. Neu khong theo thu tu nay se dinh bay "Tiền sử bệnh hiện tại"
# (thuat ngu HPI - History of Present Illness, KHONG phai isHistorical) bi
# nham thanh isHistorical chi vi chua chu "tien su".
SCOPE_CUES_ORDERED = [
    # (pattern, scope_label) - scope_label: "isHistorical" | "isFamily" | None (reset)
    (r"tiền sử bệnh hiện tại", None),
    (r"bệnh sử hiện tại", None),
    (r"triệu chứng hiện tại", None),
    (r"diễn biến bệnh", None),
    (r"lý do nhập viện", None),
    (r"đánh giá tại bệnh viện", None),
    (r"tiền sử gia đình", "isFamily"),
    (r"tiền sử bệnh nội khoa", "isHistorical"),
    (r"tiền sử ngoại khoa", "isHistorical"),
    (r"tiền sử dùng thuốc", "isHistorical"),
    (r"tiền sử bản thân", "isHistorical"),
    (r"thuốc trước nhập viện", "isHistorical"),
    (r"trước khi nhập viện", "isHistorical"),
    (r"trước nhập viện", "isHistorical"),
    (r"tiền sử", "isHistorical"),  # fallback chung chung - LUON kiem tra SAU CUNG
]

_SCOPE_PATTERN = re.compile(
    "|".join(f"(?P<g{i}>{p})" for i, (p, _) in enumerate(SCOPE_CUES_ORDERED)),
    re.IGNORECASE,
)


def find_scope_events(text: str) -> List[Tuple[int, Optional[str]]]:
    """
    Quet TOAN BO van ban 1 lan, tim cac diem "doi scope". Scope tim duoc
    ap dung cho MOI ENTITY sau vi tri do, cho den khi gap event tiep theo
    (giai quyet dung case danh sach thuoc duoi 1 header chung, khong can
    tu "tien su" xuat hien gan tung item).
    """
    events = []
    for m in _SCOPE_PATTERN.finditer(text):
        idx = next(i for i, g in enumerate(m.groups()) if g is not None)
        _, scope_label = SCOPE_CUES_ORDERED[idx]
        events.append((m.start(), scope_label))
    events.sort(key=lambda x: x[0])
    return events


def get_active_scope(events: List[Tuple[int, Optional[str]]], position: int) -> Optional[str]:
    """Scope dang active tai vi tri `position` = scope cua event GAN NHAT truoc do."""
    active = None
    for pos, scope in events:
        if pos <= position:
            active = scope
        else:
            break
    return active


# --- TANG B: SENTENCE-LEVEL LOCAL TRIGGER (NegEx-style) -------------------

NEGATION_CUES = ["không có", "không", "chưa", "phủ nhận", "loại trừ"]
FAMILY_CUES = ["bố", "mẹ", "ba", "má", "anh trai", "chị gái", "em trai", "em gái",
               "người nhà", "gia đình", "ông", "bà"]


def get_clause_boundaries(text: str, entity_start: int) -> Tuple[int, int]:
    """Tim ranh gioi menh de chua entity (lui/tien den dau ',' ';' '.' '\\n')."""
    left = entity_start
    while left > 0 and text[left - 1] not in ",;.\n":
        left -= 1
    right = entity_start
    while right < len(text) and text[right] not in ",;.\n":
        right += 1
    return left, right


def find_local_triggers(text: str, entity_start: int) -> List[str]:
    """
    Quet trigger CUC BO trong menh de chua entity (Tang B).
    QUAN TRONG: dung WORD-BOUNDARY regex (\\b), KHONG dung substring "in" -
    tieng Viet co bay kieu "không" (khong co) chua san "ông" (nguoi nha)
    nhu mot substring, gay false positive isFamily neu chi check "in".
    """
    clause_start, _ = get_clause_boundaries(text, entity_start)
    clause_before = text[clause_start:entity_start].lower()

    triggers = []
    if any(re.search(r"\b" + re.escape(cue) + r"\b", clause_before) for cue in NEGATION_CUES):
        triggers.append("isNegated")
    if any(re.search(r"\b" + re.escape(cue) + r"\b", clause_before) for cue in FAMILY_CUES):
        triggers.append("isFamily")
    return triggers


def compute_rule_based_assertions(
    text: str, entity_start: int, scope_events: List[Tuple[int, Optional[str]]]
) -> List[str]:
    """Merge Tang A (scope) + Tang B (trigger cuc bo) -> candidate assertions."""
    assertions = set()

    scope = get_active_scope(scope_events, entity_start)
    if scope in ("isHistorical", "isFamily"):
        assertions.add(scope)

    assertions.update(find_local_triggers(text, entity_start))
    return sorted(assertions)


# --- TANG C: LLM CHI BO SUNG THEM isNegated (KHONG duoc ghi de scope) ------
#
# QUAN TRONG (bai hoc tu thuc te): ban dau de LLM tu do "xac nhan/dieu chinh"
# CA 3 nhan dua tren 1 menh de cuc bo - nhung LLM 7B chi thay MENH DE (vd
# "1. amlodipine 10 mg po daily"), KHONG thay header o dau van ban da tao ra
# scope isHistorical. Vi khong co bang chung trong pham vi no thay, no tu
# XOA NHAM cac nhan isHistorical dung. => KHONG con giao cho LLM quyen ghi
# de isHistorical/isFamily nua - 2 nhan nay TIN TUYET DOI vao Tang A (scope,
# rule-based, tinh tren toan van ban). LLM chi con nhiem vu BO SUNG THEM
# isNegated (viec nay NO CO THE kiem chung duoc tu chinh menh de dua vao).

NEGATION_SYSTEM_PROMPT = """
You are a Vietnamese clinical text negation detector.

You will be given a medical entity (text + type) and the clause/sentence
containing it. Determine if the entity is NEGATED in this specific clause
(e.g. "không sốt" = no fever, entity "sốt" IS negated).

A rule-based system already checked for negation trigger words and may have
missed some (e.g. unusual phrasing). Your job is to catch cases the
rule-based system might have missed, using the clause provided.

Return ONLY a valid JSON boolean: true if negated, false otherwise. No
explanation, no markdown code fences.

Examples

Entity: "sốt" (TRIỆU_CHỨNG)
Clause: "không sốt"
Output: true

Entity: "amlodipine 10 mg po daily" (THUỐC)
Clause: "1. amlodipine 10 mg po daily"
Output: false

Entity: "ho" (TRIỆU_CHỨNG)
Clause: "bệnh nhân chưa từng ho"
Output: true
""".strip()


def classify_negation_only(entity_text: str, entity_type: str, clause_text: str) -> bool:
    """
    Goi LLM CHI de kiem tra THEM isNegated (bo sung cho rule-based Tang B,
    khong thay the). KHONG dung ham nay cho isHistorical/isFamily - 2 nhan
    do da duoc TIN TUYET DOI tu Tang A (scope, xem extract_entities()).
    """
    user_content = f'Entity: "{entity_text}" ({entity_type})\nClause: "{clause_text}"'
    messages = [
        {"role": "system", "content": NEGATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=16,
        do_sample=False,
        temperature=None,
        top_p=None,
        eos_token_id=tokenizer.eos_token_id,
    )
    raw = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[-1]:],
        skip_special_tokens=True,
    ).strip().lower()

    if "true" in raw:
        return True
    if "false" in raw:
        return False
    print(f"[WARN] Khong parse duoc negation output cho {entity_text!r}: {raw!r} - mac dinh False")
    return False


# =============================================================================
# 6. HAM CHINH: text -> list entity co position
# =============================================================================

def extract_entities(text: str, verbose: bool = False) -> List[Dict]:
    """
    Pipeline day du:
      NER (sub-problem 1):
        1. generate_candidates() - sinh candidate bang rule (khong LLM)
        2. classify_candidate() - LLM tinh chinh + phan loai tung candidate
        3. locate_refined_span() - dinh vi lai vi tri tuyet doi
      Assertion (sub-problem 3), chi ap dung cho TRIỆU_CHỨNG/CHẨN_ĐOÁN/THUỐC:
        4. find_scope_events() - quet scope toan van ban 1 LAN DUY NHAT
        5. Neu candidate la is_indication (chi dinh/ly do ke don, vd "táo bón"
           trong "...điều trị táo bón") -> assertions=[] LUON, KHONG xet gi
           them (day khong phai phat bieu doc lap ve tinh trang benh nhan).
        6. Nguoc lai: TIN TUYET DOI vao Tang A (scope) cho isHistorical/
           isFamily - KHONG cho LLM ghi de, vi LLM 7B chi thay duoc menh de
           cuc bo (khong thay header o xa), de no tu quyet dinh se BI XOA
           NHAM cac nhan dung. LLM (Tang C) chi dung de BO SUNG THEM
           isNegated cuc bo (viec nay no CO THE kiem chung tu menh de duoc
           dua vao) - hop voi ket qua rule-based Tang B, khong ghi de.
    """
    candidates = generate_candidates(text)
    if verbose:
        print(f"[Tang 1] Sinh duoc {len(candidates)} candidates:")
        for c_text, c_start, c_end, is_ind in candidates:
            print(f"  [{c_start}:{c_end}] is_indication={is_ind} {c_text!r}")
        print()

    # Quet scope 1 lan duy nhat cho ca van ban (Tang A) - khong phu thuoc
    # tung entity nen chi can tinh 1 lan, dung lai cho moi entity ben duoi.
    scope_events = find_scope_events(text)
    if verbose:
        print(f"[Assertion Tang A] Scope events: {scope_events}\n")

    results = []
    for c_text, c_start, c_end, is_indication in candidates:
        refined_list = classify_candidate(c_text)
        if not refined_list:
            if verbose:
                print(f"[Tang 2] BO QUA (khong phai entity): {c_text!r}")
            continue

        for refined in refined_list:
            start, end = locate_refined_span(text, c_start, refined["text"])
            if start is None:
                print(f"[WARN] Khong dinh vi duoc refined text {refined['text']!r} "
                      f"(candidate goc: {c_text!r}) - bo qua entity nay")
                continue

            entity_text = text[start:end]
            entity_type = refined["type"]

            assertions = []
            if entity_type in ASSERTABLE_TYPES and not is_indication:
                # CHI dung rule-based Tang A+B, KHONG qua LLM nua.
                #
                # LY DO (bang chung thuc nghiem): ban dau dung LLM
                # (classify_negation_only) de BO SUNG THEM isNegated, nhung
                # thuc te test cho thay rule-based Tang B da dung 100% (vd
                # "clonazepam 0.5 mg po qam:prn" khong co tu phu dinh nao),
                # trong khi LLM lai TU HALLUCINATE ra isNegated sai cho dung
                # case do. Tuc la o buoc nay LLM dang LAM HONG ket qua da
                # dung cua rule-based, khong mang lai loi ich nao. => bo han
                # LLM khoi buoc isNegated, chi con Tang A (scope) + Tang B
                # (trigger cuc bo, word-boundary) quyet dinh toan bo assertions.
                assertions = compute_rule_based_assertions(text, start, scope_events)
            elif is_indication and verbose:
                print(f"[Assertion] BO QUA (is_indication - chi dinh/ly do ke don): {entity_text!r}")

            results.append({
                "text": entity_text,
                "type": entity_type,
                "position": [start, end],
                "assertions": assertions,
            })
            if verbose:
                print(f"[Tang 2] [{entity_type:20s}] {entity_text!r} "
                      f"pos=[{start},{end}] assertions={assertions} "
                      f"(candidate goc: {c_text!r})")

    return results


def to_output_json(entities: List[Dict]) -> List[Dict]:
    """Format dung theo yeu cau output cua de bai (candidates ICD/RxNorm de rong)."""
    return [
        {
            "text": e["text"],
            "position": e["position"],
            "type": e["type"],
            "assertions": e.get("assertions", []),
            "candidates": [],   # TODO: sub-problem 2 (ICD/RxNorm entity linking)
        }
        for e in entities
    ]


def save_output(output: List[Dict], output_path: str):
    """Luu output ra file .json."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Da luu output vao: {output_path}")


# =============================================================================
# 6. CHAY THU
# =============================================================================

if __name__ == "__main__":
    sample_text = (
        "Danh sách thuốc trước nhập viện chính xác và đầy đủ. 1. amlodipine 10 mg po daily "
        "2. aspirin 81 mg po daily 3. metoprolol succinate xl 50 mg po daily 4. guaifenesin ml "
        "po q6h:prn điều trị ho 5. nystatin oral suspension 5 ml po qid:prn điều trị đau nhức "
        "6. acetaminophen 325-650 mg po q6h:prn điều trị sốt đau 7. pravastatin 40 mg po daily "
        "8. docusate sodium 100 mg po bid điều trị táo bón 9. senna 8.6 mg po bid:prn điều trị "
        "táo bón 10. clonazepam 0.5 mg po qam:prn điều trị lo âu 11. clonazepam 1.5 mg po qhs "
        "điều trị lo âu mất ngủ"
    )

    entities = extract_entities(sample_text, verbose=True)
    output = to_output_json(entities)

    print("\n=== OUTPUT JSON ===")
    print(json.dumps(output, ensure_ascii=False, indent=2))

    save_output(output, "output_assertv3.json")