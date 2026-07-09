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
You are an information extraction system specialized in symptom extraction.

Your task is to extract ONLY symptoms and clinical manifestations from the input text.

Each extracted entity must contain:

- text: the exact symptom span appearing in the input.
- type: always "TRIỆU_CHỨNG".

Definition

A symptom is any complaint, sign, or clinical manifestation experienced or observed in the patient.

Span Rules

Extract the longest span that represents ONE symptom.

Include:
- severity
- duration
- frequency
- quantity
- temperature
- descriptive modifiers
when they belong to the symptom.

Do NOT include:
- diagnoses
- diseases
- medications
- laboratory tests
- laboratory results
- imaging findings
- medical procedures
- physician actions
- treatment plans
- reasons for admission
- section titles

Examples

Input:
Đau bụng âm ỉ trong tháng qua

Output:
[
  {
    "text": "Đau bụng âm ỉ trong tháng qua",
    "type": "TRIỆU_CHỨNG"
  }
]

Input:
Được cho dùng levofloxacin

Output:
[]

Output Requirements

- Extract every symptom mention.
- Preserve the exact text.
- Do not normalize.
- Do not correct spelling.
- Return only valid JSON.
- Do not return explanations.
- If no symptom exists, return [].

Output format

[
  {
    "text": "...",
    "type": "TRIỆU_CHỨNG"
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
1.  Tiền sử bệnh nội khoa
    Bệnh lý mãn tính: U Sacoit tổn thương chủ yếu tại tim.

2.  Tiền sử bệnh hiện tại
    Lý do nhập viện: ho x1 ngày và kèm theo ho ra máu cỡ đồng xu x3 đêm qua
    Thời điểm khởi phát triệu chứng
    - ho x1 ngày
    - ho ra máu x3 đêm qua
    Diễn biến bệnh
    - Các triệu chứng bắt đầu với ho và mệt mỏi
    - Tối qua tiến triển thành 3 cơn ho đánh thức bệnh nhân khỏi giấc ngủ và có đờm có màu hồng
    - Sau đó 2 giờ, bệnh nhân sốt đến 38.8°C (101.8°F)
    - Không có thêm ho ra máu kể từ sáng nay
    Triệu chứng hiện tại
    - ho
    - mệt mỏi
    - ho ra máu cỡ đồng xu x3 đêm qua
    - Đờm pha màu hồng
    - fever (trở thành sốt đến 38.8°C)
    - Đau bụng  âm ỉ  trong tháng qua 
   - Các diễn biến  trước khi nhập viện
    - Đến gặp bác sĩ chăm sóc chính sáng nay
    - Được chuyển đến phòng cấp cứu khám và điều trị
    - Được cho dùng levofloxacin vì nghi ngờ viêm phế quản do viêm phổi mắc phải cộng đồng ở bệnh nhân phức tạp,  cùng tylenol

3.  Đánh giá tại bệnh viện
    Kết quả xét nghiệm: công thức máu (cbc) nâng cao lên 11.3
    Kết quả chẩn đoán hình ảnh: chụp x-quang ngực không phát hiện viêm phổi hoặc phù phổi
    Các thủ thuật đã thực hiện: Lấy mẫu cấy máu
"""

result = extract_entities(text)

print(result)