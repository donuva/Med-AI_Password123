import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
# MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

SYSTEM_PROMPT = """
You are an information extraction system specialized in medication extraction.

Your task is to extract ONLY medication mentions from the input text.

Each extracted entity must contain:

- text: the exact medication span appearing in the input.
- type: always "THUỐC".

Medication Span Rules

A medication entity may include:
- drug name
- strength
- dose
- dosage range
- formulation
- route of administration
- frequency
- administration timing
- PRN instruction if attached to the medication

The entity MUST stop immediately after the medication expression.

DO NOT include:
- indication
- diagnosis
- symptom
- disease
- treatment purpose
- reason for prescribing
- explanatory text

Examples

Input:
clonazepam 0.5 mg po qam:prn điều trị lo âu

Output:
[
  {
    "text": "clonazepam 0.5 mg po qam:prn",
    "type": "THUỐC"
  }
]

Input:
docusate sodium 100 mg po bid

Output:
[
  {
    "text": "docusate sodium 100 mg po bid",
    "type": "THUỐC"
  }
]

Output Requirements

- Extract every medication mention.
- Preserve the exact text.
- Do not normalize.
- Do not correct spelling.
- Return only valid JSON.
- Do not return explanations.
- If no medication is found, return [].

Output format

[
  {
    "text": "...",
    "type": "THUỐC"
  }
]
"""

def extract_entities(text):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": text
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        eos_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[-1]:],
        skip_special_tokens=True
    )

    return response


text = """
1'Danh sách thuốc trước nhập viện chính xác và đầy đủ. 1. amlodipine 10 mg po daily 2. aspirin 81 mg po daily 3. metoprolol succinate xl 50 mg po daily 4. guaifenesin ml po q6h:prn điều trị ho 5. nystatin oral suspension 5 ml po qid:prn điều trị đau nhức 6. acetaminophen 325-650 mg po q6h:prn điều trị sốt đau 7. pravastatin 40 mg po daily 8. docusate sodium 100 mg po bid điều trị táo bón 9. senna 8.6 mg po bid:prn điều trị táo bón 10. clonazepam 0.5 mg po qam:prn điều trị lo âu 11. clonazepam 1.5 mg po qhs điều trị lo âu mất ngủ'
"""

result = extract_entities(text)

print(result)