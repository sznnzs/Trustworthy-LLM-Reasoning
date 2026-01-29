import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

class EntropyDynamicTemperatureGenerator:
    def __init__(self, model_name, base_temperature=0.8, N=0.8, theta=1.0, device='cuda'):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_temperature = base_temperature
        self.N = N
        self.theta = theta

    def calculate_entropy_from_logits(self, logits):
        max_logits = torch.max(logits, dim=-1, keepdim=True).values
        stable_logits = logits - max_logits
        exp_logits = torch.exp(stable_logits)
        sum_exp_logits = exp_logits.sum(dim=-1, keepdim=True)
        softmax_probs = exp_logits / sum_exp_logits
        entropy = -(softmax_probs * stable_logits).sum(dim=-1)
        return entropy

    def adjust_temperature(self, entropy):
        entropy = entropy.item()
        if entropy == 0:
            return 0  # Directly use 0 to indicate greedy decoding
        return self.base_temperature * (self.N ** (self.theta / entropy))

    def generate(self, prompt, max_tokens=50):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        generated_text = prompt

        for step in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids, use_cache=True)
                logits = outputs.logits[:, -1, :]

            entropy = self.calculate_entropy_from_logits(logits)
            temperature = self.adjust_temperature(entropy)
            if temperature < 0.001:
                # Greedy decoding: Select the token with the highest probability
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # Use temperature-scaled logits to sample the next token
                scaled_logits = logits / temperature
                next_token_id = torch.multinomial(torch.softmax(scaled_logits, dim=-1), num_samples=1)

            next_token_str = self.tokenizer.decode(next_token_id.squeeze().item())
            generated_text += next_token_str
            input_ids = torch.cat([input_ids, next_token_id], dim=1)

            if next_token_str in [self.tokenizer.eos_token]:
                break

        return generated_text

    def delete(self):
        del self.model

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entropy-based dynamic temperature decoding (demo).")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Hugging Face model id or local path.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (e.g. cuda, cpu).")
    parser.add_argument("--base_temperature", type=float, default=0.8)
    parser.add_argument("--N", type=float, default=0.8)
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument(
        "--prompt",
        type=str,
        default="Answer briefly: 1 + 1 = ?",
        help="Prompt string (ignored if --prompt_file is set).",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="Path to a text file containing the prompt.",
    )
    args = parser.parse_args()

    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")

    generator = EntropyDynamicTemperatureGenerator(
        args.model_name,
        base_temperature=args.base_temperature,
        N=args.N,
        theta=args.theta,
        device=args.device,
    )

    result = generator.generate(prompt, max_tokens=args.max_tokens)
    print(result)
