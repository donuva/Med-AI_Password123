"""
Entity Linking (sub-problem 2) - goi API TRUC TIEP luc inference.

THUOC       -> RxNorm API (rxnav.nlm.nih.gov) - mien phi, KHONG can dang ky
CHAN_DOAN   -> ICD-10-CM search API (clinicaltables.nlm.nih.gov) - mien phi

Neu moi truong chay THUC SU khong co internet, cac ham nay se CATCH loi va
tra ve list rong [] thay vi crash - dam bao pipeline van chay het duoc du
entity linking khong hoat dong duoc trong truong hop do.
"""

import re
import requests
from typing import Optional


REQUEST_TIMEOUT = 5  # giay - khong doi qua lau neu mang cham/khong co mang


# =============================================================================
# THUOC -> RxNorm
# =============================================================================
#
# THIET KE (sau nhieu vong debug thuc te + tra tay qua RxNav):
#
# BAI HOC QUAN TRONG NHAT: KHONG duoc coi moi con so trong entity la
# "strength". Medication order co nhieu loai so voi Y NGHIA KHAC NHAU:
#   - strength      : "10 mg"          -> ham luong 1 don vi thuoc
#   - concentration : "0.4 MG/ML"      -> nong do (bat buoc cho dang long)
#   - dose range    : "325-650 mg"     -> khoang lieu dung (PRN), KHONG
#                      phai 1 strength duy nhat
#   - volume        : "5 ml"           -> THE TICH can uong, KHONG PHAI
#                      strength/concentration cua thuoc
#   - frequency     : "daily/bid/q6h"  -> khong lien quan RxNorm concept
#
# Neu coi volume nhu strength (vd "5 ml" cua dang long khong co concentration
# di kem) se chon nham SCD cu the trong khi dung ra phai la IN (ingredient-
# level, vi khong co nong do thi khong xac dinh duoc dung product nao).

TTY_PRIORITY = ["SCD", "SBD", "SCDC", "IN"]

ROUTE_TO_FORM_HINT = {
    "po": "Oral", "iv": "Injection", "im": "Injection", "sc": "Injection",
    "top": "Topical", "inh": "Inhalant", "pr": "Rectal", "sl": "Sublingual",
}

FORM_ONLY_WORDS = {
    "oral", "suspension", "tablet", "tablets", "capsule", "capsules",
    "solution", "cream", "ointment", "injection", "syrup", "spray",
    "xl", "er", "extended", "release", "ir",
}

BARE_UNIT_WORDS = {"mg", "ml", "mcg", "g", "unt", "unit", "units"}

_ROUTE_FREQ_PATTERN = re.compile(
    r'\b(po|iv|im|sc|top|inh|pr|sl|daily|bid|tid|qid|qhs|qam|qpm|'
    r'q\d+h|prn|stat)\b[:.]?',
    re.IGNORECASE,
)

# Thu tu kiem tra QUAN TRONG: cu the nhat (concentration) truoc, chung
# chung nhat (volume don thuan) sau cung.
_CONCENTRATION_PATTERN = re.compile(r'([\d.]+)\s*(mg|mcg|g|unt|unit)s?\s*/\s*(ml|g)\b', re.IGNORECASE)
_RANGE_PATTERN = re.compile(r'([\d.]+)\s*-\s*([\d.]+)\s*(mg|mcg|g|%)\b', re.IGNORECASE)
_STRENGTH_PATTERN = re.compile(r'([\d.]+)\s*(mg|mcg|g|%)\b', re.IGNORECASE)
_VOLUME_PATTERN = re.compile(r'([\d.]+)\s*(ml)\b', re.IGNORECASE)


def parse_drug_components(entity_text: str) -> dict:
    """
    Phan tich entity THUOC thanh cac slot co Y NGHIA RIENG BIET, tranh loi
    coi moi con so la "strength" nhu nhau:

      dose_kind = "concentration" -> dose_info = "0.4 MG/ML" (string, dung
                  truc tiep lam strength_str de loc SCD nhu binh thuong)
      dose_kind = "range"         -> dose_info = (low, high, unit) tuple
      dose_kind = "strength"      -> dose_info = "10 MG" (string)
      dose_kind = "volume_only"   -> dose_info = None (KHONG dung de loc
                  strength - day la the tich, khong phai ham luong thuoc)
      dose_kind = None            -> khong tim thay so lieu nao (vd thieu
                  hoan toan so lieu trong van ban goc)
    """
    route_match = re.search(r'\b(po|iv|im|sc|top|inh|pr|sl)\b', entity_text, re.IGNORECASE)
    route_hint = ROUTE_TO_FORM_HINT.get(route_match.group(1).lower()) if route_match else None

    cleaned = _ROUTE_FREQ_PATTERN.sub('', entity_text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip(' :,-')

    m = _CONCENTRATION_PATTERN.search(cleaned)
    if m:
        value, unit1, unit2 = m.groups()
        dose_kind, dose_info, ing_end = "concentration", f"{value} {unit1.upper()}/{unit2.upper()}", m.start()
    else:
        m = _RANGE_PATTERN.search(cleaned)
        if m:
            low, high, unit = m.groups()
            dose_kind, dose_info, ing_end = "range", (low, high, unit.upper()), m.start()
        else:
            m = _STRENGTH_PATTERN.search(cleaned)
            if m:
                value, unit = m.groups()
                dose_kind, dose_info, ing_end = "strength", f"{value} {unit.upper()}", m.start()
            else:
                m = _VOLUME_PATTERN.search(cleaned)
                if m:
                    dose_kind, dose_info, ing_end = "volume_only", None, m.start()
                else:
                    dose_kind, dose_info, ing_end = None, None, len(cleaned)

    ingredient_raw = cleaned[:ing_end].strip()
    words = ingredient_raw.split()

    # tach form-descriptor words (oral/suspension/tablet...)
    form_hints = []
    while words and words[-1].lower() in FORM_ONLY_WORDS:
        form_hints.insert(0, words.pop())
    # tach not don vi tro troi con sot lai (vd "guaifenesin ml" thieu so lieu
    # -> "ml" khong duoc bat boi cac pattern tren vi khong co so di kem)
    while words and words[-1].lower() in BARE_UNIT_WORDS:
        words.pop()

    ingredient = " ".join(words)
    if route_hint and route_hint.lower() not in [h.lower() for h in form_hints]:
        form_hints.insert(0, route_hint)

    return {
        "ingredient": ingredient,
        "form_hints": form_hints,
        "dose_kind": dose_kind,
        "dose_info": dose_info,
    }


def _get_drugs_concept_groups(ingredient: str) -> list:
    """Goi getDrugs API, tra ve raw conceptGroup list (co the rong)."""
    resp = requests.get(
        "https://rxnav.nlm.nih.gov/REST/drugs.json",
        params={"name": ingredient},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json().get("drugGroup", {}).get("conceptGroup", []) or []


def _filter_by_strength_and_form(props: list, strength_str: str, form_hints: list) -> list:
    """Loc props theo substring strength, roi loc tiep theo form_hints neu con mo ho."""
    if strength_str:
        matches = [p for p in props if strength_str.lower() in p.get("name", "").lower()]
    else:
        matches = list(props)
    if len(matches) > 1 and form_hints:
        for hint in form_hints:
            hint_matches = [p for p in matches if hint.lower() in p.get("name", "").lower()]
            if hint_matches:
                matches = hint_matches
                break
    return matches


def _extract_strength_value(name: str) -> Optional[float]:
    """Lay gia tri so dau tien trong ten concept (vd 'clonazepam 1 MG...' -> 1.0)."""
    m = _STRENGTH_PATTERN.search(name)
    return float(m.group(1)) if m else None


def _find_nearest_available_strength(props: list, target_value: float) -> list:
    """
    Khi khong co SCD nao dung strength yeu cau (vd '1.5 MG' khong phai
    ham luong thuong mai co that), tim SCD co strength GAN NHAT trong danh
    sach cac SCD hien co cung ingredient (vd 1 MG hoac 2 MG).
    """
    candidates_with_value = []
    for p in props:
        val = _extract_strength_value(p.get("name", ""))
        if val is not None:
            candidates_with_value.append((abs(val - target_value), p))
    if not candidates_with_value:
        return []
    candidates_with_value.sort(key=lambda x: x[0])
    best_diff = candidates_with_value[0][0]
    return [p for diff, p in candidates_with_value if diff == best_diff]


def rxnorm_lookup(entity_text: str, max_candidates: int = 1, disambiguator=None) -> list:
    """
    RxNorm lookup phan biet ro strength/concentration/range/volume (xem
    parse_drug_components). Xu ly theo tung dose_kind:

      "concentration" -> loc SCD theo concentration nhu strength binh thuong
      "strength"      -> loc SCD theo strength; neu KHONG co SCD nao khop
                         (vd lieu khong ton tai thuong mai nhu "1.5 MG") ->
                         tim SCD co strength GAN NHAT (vd "1 MG") thay vi
                         chiu thua ve fuzzy match
      "range"         -> thu LOWER-BOUND truoc (vd "325 MG"), neu khong
                         co thi thu UPPER-BOUND (vd "650 MG")
      "volume_only"   -> KHONG loc theo the tich (vi the tich khong phai
                         strength) - lay thang IN-level (ingredient don
                         thuan, vi thieu concentration thi khong the xac
                         dinh dung product nao)
      None (thieu so lieu hoan toan) -> lay SCD dau tien tim duoc qua
                         getDrugs (best-guess, chap nhan co the sai vi
                         van ban goc thieu thong tin can thiet)

    disambiguator: ham optional (entity_text, candidates) -> rxcui|None,
    CHI goi khi hard-filter van con >1 candidate mo ho. Neu tra ve gia tri
    khong nam trong candidates da cho -> tu dong bo qua (khong tin
    hallucination), giu nguyen candidate dau tien.
    """
    slots = parse_drug_components(entity_text)
    ingredient = slots["ingredient"]
    form_hints = slots["form_hints"]
    dose_kind = slots["dose_kind"]
    dose_info = slots["dose_info"]

    if not ingredient:
        return []

    try:
        concept_groups = _get_drugs_concept_groups(ingredient)

        # ===== CASE: volume_only -> KHONG loc strength, ve thang IN =====
        if dose_kind == "volume_only":
            in_group = next((g for g in concept_groups if g.get("tty") == "IN"), None)
            if in_group and in_group.get("conceptProperties"):
                return [in_group["conceptProperties"][0]["rxcui"]]
            # getDrugs khong tra IN -> tra IN rieng qua rxcui.json
            return _rxnorm_ingredient_only_lookup(ingredient, max_candidates)

        # ===== CASE: range -> thu lower-bound truoc, upper-bound sau =====
        if dose_kind == "range":
            low, high, unit = dose_info
            for value in (low, high):
                strength_str = f"{value} {unit}"
                result = _search_scd_sbd(concept_groups, strength_str, form_hints,
                                          entity_text, disambiguator, max_candidates)
                if result:
                    return result
            # khong tim thay ca 2 dau mut -> fallback fuzzy
            return _rxnorm_approximate_fallback(entity_text, max_candidates)

        # ===== CASE: concentration hoac strength don =====
        if dose_kind in ("concentration", "strength"):
            strength_str = dose_info
            result = _search_scd_sbd(concept_groups, strength_str, form_hints,
                                      entity_text, disambiguator, max_candidates)
            if result:
                return result

            # "strength" khong co SCD nao khop CHINH XAC -> co the la lieu
            # khong ton tai thuong mai (vd "1.5 MG") -> tim strength GAN NHAT
            if dose_kind == "strength":
                target_value = _extract_strength_value(strength_str)
                scd_group = next((g for g in concept_groups if g.get("tty") == "SCD"), None)
                if scd_group and target_value is not None:
                    nearest = _find_nearest_available_strength(
                        scd_group.get("conceptProperties", []) or [], target_value
                    )
                    nearest = [p for p in nearest if
                               not form_hints or any(h.lower() in p.get("name", "").lower() for h in form_hints)] or nearest
                    if nearest:
                        return [p["rxcui"] for p in nearest[:max_candidates]]

            # SCDC rieng (getDrugs khong tra SCDC)
            scdc_result = _rxnorm_scdc_lookup(ingredient, dose_info, max_candidates)
            if scdc_result:
                return scdc_result

            return _rxnorm_approximate_fallback(entity_text, max_candidates)

        # ===== CASE: khong co so lieu nao (thieu hoan toan trong van ban goc) =====
        # Best-guess: lay SCD dau tien tim duoc. Day la gioi han thuc su
        # (van ban goc thieu thong tin can thiet de xac dinh chinh xac),
        # KHONG hard-code dap an rieng cho tung thuoc cu the.
        for tty in TTY_PRIORITY:
            group = next((g for g in concept_groups if g.get("tty") == tty), None)
            if group and group.get("conceptProperties"):
                return [group["conceptProperties"][0]["rxcui"]]
        return _rxnorm_approximate_fallback(entity_text, max_candidates)

    except (requests.RequestException, KeyError, ValueError) as e:
        print(f"[WARN] RxNorm lookup that bai cho {entity_text!r}: {e}")
        return _rxnorm_approximate_fallback(entity_text, max_candidates)


def _search_scd_sbd(concept_groups, strength_str, form_hints, entity_text, disambiguator, max_candidates):
    """Tim trong SCD/SBD (getDrugs) theo strength_str + form_hints, co disambiguator."""
    for tty in ("SCD", "SBD"):
        group = next((g for g in concept_groups if g.get("tty") == tty), None)
        if not group:
            continue
        matches = _filter_by_strength_and_form(
            group.get("conceptProperties", []) or [], strength_str, form_hints
        )
        if not matches:
            continue

        if len(matches) > 1 and disambiguator is not None:
            candidate_dicts = [
                {"rxcui": p["rxcui"], "name": p.get("name", ""), "tty": tty} for p in matches
            ]
            chosen_rxcui = disambiguator(entity_text, candidate_dicts)
            valid_rxcuis = {c["rxcui"] for c in candidate_dicts}
            if chosen_rxcui in valid_rxcuis:
                matches = [p for p in matches if p["rxcui"] == chosen_rxcui] + \
                          [p for p in matches if p["rxcui"] != chosen_rxcui]

        return [p["rxcui"] for p in matches[:max_candidates]]
    return None


def _rxnorm_ingredient_only_lookup(ingredient: str, max_candidates: int) -> list:
    """Tra rieng IN-level (getDrugs khong tra IN trong conceptGroup)."""
    try:
        resp = requests.get(
            "https://rxnav.nlm.nih.gov/REST/rxcui.json",
            params={"name": ingredient, "search": 2},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        ids = data.get("idGroup", {}).get("rxnormId", [])
        return ids[:max_candidates]
    except (requests.RequestException, KeyError, ValueError):
        return []


def _rxnorm_scdc_lookup(ingredient: str, strength_str: str, max_candidates: int) -> list:
    """Tra rieng SCDC (getDrugs khong tra SCDC), dung ten dang chuan hoa."""
    canonical = f"{' '.join(w.capitalize() for w in ingredient.split())} {strength_str}"
    try:
        resp = requests.get(
            "https://rxnav.nlm.nih.gov/REST/rxcui.json",
            params={"name": canonical, "search": 2},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        ids = data.get("idGroup", {}).get("rxnormId", [])
        return ids[:max_candidates]
    except (requests.RequestException, KeyError, ValueError):
        return []


def _rxnorm_approximate_fallback(entity_text: str, max_candidates: int) -> list:
    """Fallback fuzzy match cuoi cung (vd ten sai chinh ta, du lieu qua bat thuong)."""
    cleaned = _ROUTE_FREQ_PATTERN.sub('', entity_text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip(' :,-')
    if not cleaned:
        return []
    try:
        resp = requests.get(
            "https://rxnav.nlm.nih.gov/REST/approximateTerm.json",
            params={"term": cleaned, "maxEntries": max_candidates * 3},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("approximateGroup", {}).get("candidate", [])
        seen = set()
        rxcuis = []
        for c in candidates:
            rxcui = c.get("rxcui")
            if rxcui and rxcui not in seen:
                seen.add(rxcui)
                rxcuis.append(rxcui)
        return rxcuis[:max_candidates]
    except (requests.RequestException, KeyError, ValueError) as e:
        print(f"[WARN] RxNorm approximateTerm fallback that bai cho {entity_text!r}: {e}")
        return []


# =============================================================================
# CHAN_DOAN -> ICD-10-CM
# =============================================================================

def icd10_lookup(diagnosis_text_english: str, max_candidates: int = 1) -> list:
    """
    Goi NLM Clinical Tables API de tim ma ICD-10-CM candidate cho 1
    CHAN_DOAN (input phai la tieng Anh - xem translate_diagnosis_to_english
    trong pipeline chinh de dich truoc khi goi ham nay).

    LUU Y ve max_candidates=1 (mac dinh, theo yeu cau fix cung): de bai
    goc co the chap nhan nhieu ma cho 1 chan doan mo ho (vd K21.0/K21.9
    cho GERD), nhung dua tren quan sat thuc te candidate co xu huong
    tra ve nhieu ma khong lien quan/qua rong (vd F10.94/F10.980/F10.982
    cho "hoi chung nghien ruou" trong khi chi can 1 ma dai dien), nen
    gioi han cung xuong 1 de tranh bi tru diem do du thua/sai candidate.

    QUAN TRONG: API cua NLM dung thuat toan autocomplete/fuzzy/stemming
    noi bo - da phat hien case query "vascular disease" tra ve nham ca ma
    "vaso-occlusive" (vi "vaso-" va "vascul-" cung goc Latin "vas" nen bi
    stemmer coi la lien quan). De tranh loi nay, ta LAY DU candidate hon
    can (maxList*5) roi TU LOC LAI phia client: chi giu candidate ma ten
    (name) chua DAY DU tung tu khoa chinh trong cau query (khop nguyen
    tu theo word-boundary, khong dua vao fuzzy/stemming cua API nua).
    Neu loc xong khong con candidate nao (qua nghiem ngat), fallback ve
    ket qua tho ban dau (thay vi tra rong).

    Tra ve list cac ma ICD-10 (string), toi da `max_candidates`.
    """
    if not diagnosis_text_english:
        return []

    # tu ngan (<=2 ky tu) thuong la gioi tu/mao tu, khong dung de loc
    query_words = [w for w in re.findall(r"[a-zA-Z]+", diagnosis_text_english) if len(w) > 2]

    try:
        resp = requests.get(
            "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search",
            params={
                "sf": "code,name",
                "terms": diagnosis_text_english,
                "maxList": max_candidates * 5,  # lay du hon truoc khi tu loc lai
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        code_name_pairs = data[3] if len(data) > 3 else []

        if not query_words:
            return [pair[0] for pair in code_name_pairs[:max_candidates]]

        # loc lai: CHI giu candidate co TEN chua DU TAT CA tu khoa chinh
        # (word-boundary, khong phai substring tho - tranh "vas" khop nham
        # ca "vascular" lan "vaso")
        filtered = []
        for code, name in code_name_pairs:
            name_lower = name.lower()
            if all(re.search(r'\b' + re.escape(w.lower()) + r'\b', name_lower) for w in query_words):
                filtered.append(code)

        if filtered:
            return filtered[:max_candidates]

        # loc qua nghiem ngat khong con gi -> fallback ve ket qua tho ban dau
        print(f"[WARN] ICD-10 loc nghiem ngat khong con candidate nao cho "
              f"{diagnosis_text_english!r}, dung ket qua tho khong loc")
        return [pair[0] for pair in code_name_pairs[:max_candidates]]

    except (requests.RequestException, IndexError, ValueError) as e:
        print(f"[WARN] ICD-10 lookup that bai cho {diagnosis_text_english!r}: {e}")
        return []


if __name__ == "__main__":
    # Test nhanh (can internet that su de chay, sandbox chuan bi code nay
    # khong co quyen truy cap domain nlm.nih.gov nen khong tu test duoc)
    print("=== Test RxNorm (getDrugs + loc theo strength/TTY) ===")
    for drug in [
        "amlodipine 10 mg po daily",
        "aspirin 81 mg po daily",
        "docusate sodium 100 mg po bid",
        "nystatin oral suspension 5 ml po qid:prn",
    ]:
        print(f"{drug!r} -> {rxnorm_lookup(drug)}")

    print("\n=== Test ICD-10 ===")
    for dx in ["gastroesophageal reflux disease", "asthma"]:
        print(f"{dx!r} -> {icd10_lookup(dx)}")