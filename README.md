# 🧠 TinyLlama-MedQA: Lightweight Medical Q&A Chat Model

Fine-tuned version of **TinyLlama-1.1B-Chat** on a curated dataset of medical instructions and answers. Optimized for **medical question answering** tasks using **LoRA** and **PEFT** with efficient 4-bit quantized training. Ideal for educational, research, and prototype-level medical assistants.

---

## 🚀 Project Overview

| Model              | Base: TinyLlama-1.1B-Chat v1.0 |
|--------------------|-------------------------------|
| Fine-Tuning Method | PEFT + LoRA (8-bit adapters)  |
| Dataset            | 5,000 Medical QA Instructions |
| Format             | Instruction + Response         |
| Usage              | Educational, Research, Demo   |

---

### Instruction:
What are the symptoms of diabetes?

### Response:
The symptoms of diabetes can include increased thirst, frequent urination, fatigue, blurred vision, and unexplained weight loss. In some cases, people may experience slow-healing sores or frequent infections. It is important to consult a healthcare provider for accurate diagnosis and treatment.

## 📦 Features

- 🩺 Instruction-tuned on 5k+ medical Q&A samples  
- 🧠 Based on TinyLlama: compact 1.1B model  
- 🧪 Efficient LoRA adapters (only adapter weights saved)  
- ⚡ Fast inference with 4-bit quantization  
- 🛠️ Ideal for MedChat, educational bots, and prompt experiments

---

## Intended Use

✅ Educational tools  
✅ Healthcare chatbots (non-diagnostic)  
✅ Research on instruction-tuning in medical NLP  
✅ Prototype demos with LLMs in medicine  

❌ Not suitable for clinical diagnosis or emergency response  

---

##  Limitations & Warnings

- May hallucinate medical facts  
- Not fine-tuned with real-world patient data  
- Designed for low-risk use only  
- Ensure human expert verification before deployment  

---

## Environmental Impact

- **Hardware**: Google Colab with T4 GPU  
- **Training Time**: ~20 minutes  
- **Estimated Emissions**: < 0.5 kg CO₂e  

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

## 🛠️ Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "your-username/tinyllama-medqa"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "### Instruction:\nWhat are the symptoms of diabetes?\n\n### Response:\n"

