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

text = """Danh sách thuốc trước nhập viện chính xác và đầy đủ. 1. amlodipine 10 mg po daily 2. aspirin 81 mg po daily 3. metoprolol succinate xl 50 mg po daily 4. guaifenesin ml po q6h:prn điều trị ho 5. nystatin oral suspension 5 ml po qid:prn điều trị đau nhức 6. acetaminophen 325-650 mg po q6h:prn điều trị sốt đau 7. pravastatin 40 mg po daily 8. docusate sodium 100 mg po bid điều trị táo bón 9. senna 8.6 mg po bid:prn điều trị táo bón 10. clonazepam 0.5 mg po qam:prn điều trị lo âu 11. clonazepam 1.5 mg po qhs điều trị lo âu mất ngủ"""


from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="unsloth/gemma-4-31B-it-GGUF",
	filename="gemma-4-31B-it-UD-Q8_K_XL.gguf",
    n_ctx=4096,
    n_threads=8,
    n_gpu_layers=-1,
    verbose=False,
    cache=True
)




USER_PROMPT = f"""
Input:

{text}

Hãy trả về JSON.
"""

messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT,
    },
    {
        "role": "user",
        "content": USER_PROMPT,
    },
]

output = llm.create_chat_completion(
    messages=messages,
    temperature=0,
    top_p=1,
    response_format={"type": "json_object"},
)


def add_positions(text, entities):
    used = []

    for entity in entities:
        phrase = entity["text"]

        start = 0
        while True:
            idx = text.find(phrase, start)
            if idx == -1:
                entity["position"] = [-1, -1]
                break

            end = idx + len(phrase) - 1

            overlap = False
            for s, e in used:
                if not (end < s or idx > e):
                    overlap = True
                    break

            if overlap:
                start = idx + 1
                continue

            entity["position"] = [idx, end]
            used.append((idx, end))
            break

    return entities



result = output["choices"][0]["message"]["content"]
print(result)


import json

data = json.loads(result)

if isinstance(data, dict):
    entities = data["entities"]
else:
    entities = data
entities = add_positions(text, entities)

print(json.dumps(entities, ensure_ascii=False, indent=2))