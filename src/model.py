import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from src.config import MODEL_NAME, HUGGINGFACE_TOKEN


def load_model_4bit(model_name: str = None):
    """Load LLM with 4-bit quantization (NF4 + double quant).

    Works on RTX 4060 (8GB VRAM) with 7-8B parameter models.
    """
    model_name = model_name or MODEL_NAME

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=HUGGINGFACE_TOKEN,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=HUGGINGFACE_TOKEN,
    )

    print(f"Model loaded: {model_name}")
    print(f"Device map: {model.hf_device_map}")
    return tokenizer, model
