from transformers.pipelines import pipeline

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def load_model():
    return pipeline(
        "text-generation",
        model=MODEL_NAME,
        device="cpu",
    )

def generate_response(pipe, messages, gen_params):
    input_text = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    result = pipe(
        input_text,
        **gen_params,
        return_full_text=False,
        eos_token_id=pipe.tokenizer.eos_token_id,
        pad_token_id=pipe.tokenizer.eos_token_id,
    )
    generated_text = result[0].get("generated_text") or result[0].get("text", "")
    if "<|assistant|>" in generated_text:
        assistant_part = generated_text.split("<|assistant|>", 1)[1]
        response = assistant_part.split("<|", 1)[0].strip()
    else:
        response = generated_text.strip()
    return response or "No response generated."

pipe = load_model()

messages = [
    {"role": "system", "content": "You are a helpful assistant. Be concise and factual."},
    {"role": "user", "content": "Tell me about langchain"},
]

gen_params = {
    "max_new_tokens": 64,
    "do_sample": False,
    "repetition_penalty": 1.2,
    "no_repeat_ngram_size": 3,
}

response = generate_response(pipe, messages, gen_params)
messages.append({"role": "assistant", "content": response})
print(response)
