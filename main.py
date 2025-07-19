from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = FastAPI(
    title="Multilingual Translator API",
    description="Translate between languages using the NLLB-200 model.",
    version="1.0.0"
)

# Load translation model
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
if torch.cuda.is_available():
    model = model.to("cuda")

class TranslateRequest(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str

class TranslateResponse(BaseModel):
    translation: str

@app.post("/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest):
    inputs = tokenizer(req.text, return_tensors="pt").to(model.device)
    inputs["forced_bos_token_id"] = tokenizer.lang_code_to_id[req.tgt_lang]
    outputs = model.generate(**inputs)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return TranslateResponse(translation=translation)
