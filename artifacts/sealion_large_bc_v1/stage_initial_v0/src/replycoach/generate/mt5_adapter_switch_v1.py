"""Independent task adapters sharing one frozen, adapted mT5 checkpoint."""

from peft import LoraConfig, PeftModel, get_peft_model


def task_adapter_config():
    return LoraConfig(
        task_type="SEQ_2_SEQ_LM", r=16, lora_alpha=32,
        lora_dropout=0.1, target_modules=["q", "v"],
    )


def new_task_adapter(base):
    """Create a fresh adapter; never continue another task's adapter."""
    model = get_peft_model(base, task_adapter_config())
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable):
        raise ValueError("Only the new LoRA adapter may be trainable")
    return model


def activate_inference(model, name):
    """PEFT can make a selected adapter trainable; explicitly freeze inference."""
    if name not in model.peft_config:
        raise ValueError(f"Unknown adapter: {name}")
    model.set_adapter(name)
    model.requires_grad_(False)
    model.eval()
    if list(model.active_adapters) != [name]:
        raise ValueError("Exactly one task adapter must be active")
    return model


def load_switching_adapters(base, english_path, singlish_path):
    model = PeftModel.from_pretrained(
        base, english_path, adapter_name="english", is_trainable=False,
    )
    model.load_adapter(singlish_path, adapter_name="singlish", is_trainable=False)
    return activate_inference(model, "english")
