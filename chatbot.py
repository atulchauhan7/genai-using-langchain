
# Local LLM chat using transformers pipeline (TinyLlama)
from transformers.pipelines import pipeline

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def load_model():
    return pipeline(
        "text-generation",
        model=MODEL_NAME,
        device="cpu",
    )

def get_response(pipe, user_input: str) -> str:
    messages = [{"role": "user", "content": user_input}]
    input_text = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    result = pipe(input_text, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.95, repetition_penalty=1.1)
    generated_text = result[0].get("generated_text") or result[0].get("text", "")
    if "<|assistant|>" in generated_text:
        assistant_part = generated_text.split("<|assistant|>", 1)[1]
        response = assistant_part.split("<|", 1)[0].strip()
    else:
        response = generated_text[len(input_text):].strip()
    return response or "No response generated."

pipe = load_model()

print("Local LLM Chatbot (TinyLlama) - type 'exit' or 'quit' to stop.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chat.")
        break
    response = get_response(pipe, user_input)
    print("AI:", response)