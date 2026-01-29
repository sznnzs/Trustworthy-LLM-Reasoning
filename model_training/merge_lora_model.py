import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main(base_model_name: str, lora_path: str, merged_model_path: str) -> None:
    """Merge a LoRA adapter into the base model and save a standalone checkpoint."""

    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model.resize_token_embeddings(len(tokenizer))

    peft_model = PeftModel.from_pretrained(model, lora_path)
    peft_model = peft_model.merge_and_unload()

    peft_model.save_pretrained(merged_model_path)
    tokenizer.save_pretrained(merged_model_path)
    print(f"Merged model and tokenizer saved to {merged_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA weights into a base model and save the merged model.")
    parser.add_argument("--base_model_name", type=str, required=True, help="Base model (HF id or local path).")
    parser.add_argument("--lora_path", type=str, required=True, help="LoRA checkpoint path.")
    parser.add_argument("--merged_model_path", type=str, required=True, help="Output path for the merged model.")

    args = parser.parse_args()
    main(args.base_model_name, args.lora_path, args.merged_model_path)
