import json
import re
from pathlib import Path

from tqdm import tqdm
from llama_cpp import Llama


SYSTEM_PROMPT = """
You are a medical named entity recognition (NER) system.

Your task is to extract ALL medical entities from a Vietnamese medical document.

Return ONLY a valid JSON array.

Each entity must have the following format:

{
    "text": "...",
    "type": "...",
    "assertions": []
}

The allowed entity types are:

- TRIỆU_CHỨNG
- CHẨN_ĐOÁN
- THUỐC
- TÊN_XÉT_NGHIỆM
- KẾT_QUẢ_XÉT_NGHIỆM

Definitions:

- TRIỆU_CHỨNG: Signs or symptoms experienced by the patient.
- CHẨN_ĐOÁN: Diseases or diagnoses made by a clinician.
- THUỐC: Medication names, including dosage or formulation if present.
- TÊN_XÉT_NGHIỆM: Laboratory test names or laboratory markers.
- KẾT_QUẢ_XÉT_NGHIỆM: Laboratory test values, including units if present.

Note: For THUỐC, extract the complete medication mention exactly as it appears in the text, including dosage, strength, formulation, route, and administration frequency whenever they are present.

The "assertions" field is only applicable to:

- TRIỆU_CHỨNG
- CHẨN_ĐOÁN
- THUỐC

Allowed assertion values are:

- "isNegated"
    The entity is explicitly negated.
    Examples:
    - không ho
    - chưa sốt
    - phủ nhận đau ngực

- "isFamily"
    The entity refers to a family member rather than the patient.
    Examples:
    - bố bị tăng huyết áp
    - mẹ mắc đái tháo đường

- "isHistorical"
    The entity belongs to the patient's past medical history or previous medication history.
    Examples:
    - tiền sử hen phế quản
    - đã từng bị viêm phổi
    - đang dùng thuốc trước nhập viện
    - danh sách thuốc trước nhập viện

If none of the assertions apply, return:

"assertions": []

For entity types TÊN_XÉT_NGHIỆM and KẾT_QUẢ_XÉT_NGHIỆM, always return:

"assertions": []

Requirements:

- Extract every medical entity appearing in the text.
- Do not invent entities.
- Preserve the original text exactly as it appears in the input.
- Do not normalize or correct spelling.
- Do not include position or candidate codes.
- Do not output explanations or markdown.
- Output ONLY a valid JSON array.
"""


INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


llm = Llama.from_pretrained(
    repo_id="unsloth/gemma-4-31B-it-GGUF",
    filename="gemma-4-31B-it-UD-Q8_K_XL.gguf",
    n_ctx=4096,
    n_gpu_layers=-1,
    n_threads=8,
    temperature=0,
    verbose=False,
)


def add_positions(text, entities):
    """
    Add character positions to extracted entities.
    """

    occupied = []

    for entity in entities:
        phrase = entity["text"]

        search_start = 0

        while True:
            idx = text.find(phrase, search_start)

            if idx == -1:
                entity["position"] = [-1, -1]
                break

            end = idx + len(phrase) - 1

            overlap = any(
                not (end < s or idx > e)
                for s, e in occupied
            )

            if overlap:
                search_start = idx + 1
                continue

            entity["position"] = [idx, end]
            occupied.append((idx, end))
            break

    return entities


def clean_json(text):
    text = text.strip()

    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    return text


def infer(text):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": f"Input:\n\n{text}",
        },
    ]

    output = llm.create_chat_completion(
        messages=messages,
        temperature=0,
        top_p=1,
        response_format={"type": "json_object"},
    )

    result = output["choices"][0]["message"]["content"]

    result = clean_json(result)

    data = json.loads(result)

    if isinstance(data, dict):
        entities = data.get("entities", [])
    elif isinstance(data, list):
        entities = data
    else:
        raise RuntimeError("Unexpected model output.")

    entities = add_positions(text, entities)

    return entities


def main():
    txt_files = sorted(INPUT_DIR.glob("*.txt"))

    print(f"Found {len(txt_files)} files")

    for txt_file in tqdm(txt_files):

        text = txt_file.read_text(encoding="utf-8").strip()

        try:
            entities = infer(text)

            output_path = OUTPUT_DIR / f"{txt_file.stem}.json"

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    entities,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        except Exception as e:
            print(f"[ERROR] {txt_file.name}: {e}")


if __name__ == "__main__":
    main()