from transformers import AutoModelForCausalLM, AutoTokenizer

class LoRAModel:
    def __init__(self, model_name="huggingface/gemini-lora"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_answer(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=500)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

