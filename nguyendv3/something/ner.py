"""
Vietnamese Medical NER Pipeline (v2 - Candidate Generation + LLM Classification)
---------------------------------------------------------------------------------
Kien truc 2 tang, giai quyet van de LLM "bo cuoc som" khi phai trich xuat
nguyen mot van ban dai co nhieu muc liet ke:

  TANG 1 (rule-based, KHONG dung LLM):
      generate_candidates() - sinh danh sach candidate span bang cach cat
      van ban theo cac boundary co the doan truoc (so thu tu, tu khoa
      "dieu tri", dau cau...). Uu tien KHONG BO SOT (candidate co the gom
      thua tu ngu canh, se duoc LLM tinh chinh o tang 2).

  TANG 2 (LLM, pham vi nho - moi lan chi xu ly 1 candidate ngan):
      Voi moi candidate, LLM chi lam 2 viec nhe:
        (a) tinh chinh lai dung ranh gioi entity (bo tu ngu canh thua nhu
            "duoc chan doan mac benh", "co tien su su dung"...)
        (b) phan loai type, hoac tra ve "KHONG_PHAI" neu candidate khong
            chua khai niem y te nao (vd cau mo dau "Danh sach thuoc...")
      Vi pham vi rat nho (1 cum tu ngan), LLM 7B xu ly de dang va dang tin
      cay hon nhieu so voi bat no trich xuat toan bo van ban dai 1 luc.

Sau khi co refined text, vi tri tuyet doi trong van ban goc duoc tinh lai
bang string search TRONG PHAM VI candidate (khong phai toan van ban), nen
do chinh xac gan nhu tuyet doi.

Assertions (isNegated/isFamily/isHistorical) va candidates (ICD/RxNorm)
CHUA lam trong file nay - se bo sung sau (sub-problem 2 va 3 rieng).
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
    r'(?:(?<=\s)|^)\d{1,2}\.\s+'    # so thu tu "1. " "10. " (co khoang trang sau,
                                     # tranh nham voi so thap phan "0.5")
    r'|\s*điều trị\s+'              # tu khoa "dieu tri" (phan cach thuoc/chi dinh)
    r'|,\s*(?!\d)'                   # dau phay KHONG nam giua 2 chu so (tranh cat "14,43")
    r'|\.(?!\d)\s*'                  # dau cham KHONG theo sau boi chu so (tranh cat "0.5")
    r'|;\s*'                         # dau cham phay
    r'|:\s*(?=\d)'                   # dau hai cham CHI KHI theo sau la chu so
                                     # (tach ten xet nghiem/ket qua, khong cat "q6h:prn")
)


def generate_candidates(text: str) -> List[Tuple[str, int, int]]:
    """
    Sinh candidate spans bang rule-based segmentation.
    Tra ve list (candidate_text, start, end) - start/end la vi tri TUYET DOI
    trong van ban goc.
    """
    boundaries = list(_BOUNDARY_PATTERN.finditer(text))

    fragments = []
    pos = 0
    for b in boundaries:
        if b.start() > pos:
            fragments.append((pos, b.start()))
        pos = b.end()
    if pos < len(text):
        fragments.append((pos, len(text)))

    candidates = []
    for start, end in fragments:
        raw = text[start:end]
        stripped = raw.strip()
        if not stripped:
            continue
        lstrip_len = len(raw) - len(raw.lstrip())
        real_start = start + lstrip_len
        real_end = real_start + len(stripped)
        candidates.append((stripped, real_start, real_end))

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
   character-for-character - do NOT paraphrase, do NOT add/remove
   characters, do NOT merge two distinct symptoms/concepts into one span
   even if they appear with no punctuation between them.
   Strip away any connector/descriptive words that are not part of the
   entity itself.
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

Input phrase: "14,43"
Output: [{{"text": "14,43", "type": "KẾT_QUẢ_XÉT_NGHIỆM"}}]

Input phrase: "lo âu mất ngủ"
Output: [{{"text": "lo âu", "type": "TRIỆU_CHỨNG"}}, {{"text": "mất ngủ", "type": "TRIỆU_CHỨNG"}}]

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
# 5. HAM CHINH: text -> list entity co position
# =============================================================================

def extract_entities(text: str, verbose: bool = False) -> List[Dict]:
    """
    Pipeline day du 2 tang:
      1. generate_candidates() - sinh candidate bang rule (khong LLM)
      2. classify_candidate() - LLM tinh chinh + phan loai tung candidate
      3. locate_refined_span() - dinh vi lai vi tri tuyet doi
    """
    candidates = generate_candidates(text)
    if verbose:
        print(f"[Tang 1] Sinh duoc {len(candidates)} candidates:")
        for c_text, c_start, c_end in candidates:
            print(f"  [{c_start}:{c_end}] {c_text!r}")
        print()

    results = []
    for c_text, c_start, c_end in candidates:
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

            results.append({
                "text": text[start:end],
                "type": refined["type"],
                "position": [start, end],
            })
            if verbose:
                print(f"[Tang 2] [{refined['type']:20s}] {text[start:end]!r} "
                      f"pos=[{start},{end}]  (candidate goc: {c_text!r})")

    return results


def to_output_json(entities: List[Dict]) -> List[Dict]:
    """Format dung theo yeu cau output cua de bai (assertions/candidates de rong)."""
    return [
        {
            "text": e["text"],
            "position": e["position"],
            "type": e["type"],
            "assertions": [],   # TODO: sub-problem 3
            "candidates": [],   # TODO: sub-problem 2
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

    save_output(output, "output.json")