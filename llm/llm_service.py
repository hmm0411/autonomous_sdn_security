import os
from huggingface_hub import InferenceClient

# lấy token từ môi trường
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN not found in environment variables")

client = InferenceClient(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    token=HF_TOKEN
)

def call_llm(prompt):
    try:
        res = client.chat_completion(
            messages=[
                {"role": "system", "content": "You are an SDN network security expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3,
            top_p=0.9
        )

        return res.choices[0].message.content

    except Exception as e:
        return f"LLM_ERROR: {str(e)}"