
# Local LLM chat using transformers pipeline (TinyLlama)
import os
from transformers.pipelines import pipeline
from transformers import StoppingCriteria, StoppingCriteriaList
import re

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def load_model():
    return pipeline(
        "text-generation",
        model=MODEL_NAME,
        device="cpu",
    )

class StopAfterFirstSentence(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.start_len = None
        self._pattern = re.compile(r"[.!?](?:\s|\Z)")

    def __call__(self, input_ids, scores, **kwargs):
        seq = input_ids[0]
        if self.start_len is None:
            self.start_len = seq.shape[-1]
            return False
        gen_ids = seq[self.start_len:]
        if gen_ids.numel() == 0:
            return False
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return bool(self._pattern.search(text))

def get_response(pipe, history, user_input: str, max_turns: int = 3) -> str:
    # Only add the user message before generation, but limit context to recent turns
    # Always include the system prompt (history[0])
    turns = history[1:]
    if len(turns) > max_turns * 2:
        turns = turns[-max_turns*2:]
    temp_history = [history[0]] + turns + [{"role": "user", "content": user_input}]
    input_text = pipe.tokenizer.apply_chat_template(
        temp_history, tokenize=False, add_generation_prompt=True
    )
    stopping = StoppingCriteriaList([StopAfterFirstSentence(pipe.tokenizer)])
    result = pipe(
        input_text,
        max_new_tokens=24,
        do_sample=False,
        repetition_penalty=1.3,
        no_repeat_ngram_size=4,
        return_full_text=False,
        eos_token_id=pipe.tokenizer.eos_token_id,
        pad_token_id=pipe.tokenizer.eos_token_id,
        stopping_criteria=stopping,
    )
    generated_text = result[0].get("generated_text") or result[0].get("text", "")
    response = generated_text.strip()
    # Now update the real history with both user and assistant
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response})
    return response or "No response generated."


os.environ["TRANSFORMERS_VERBOSITY"] = "error"

pipe = load_model()
# System prompt + minimal few-shot to steer succinct answers (model-driven)
history = [
    {
        "role": "system",
        "content": (
            "You are a concise, factual assistant. Answer ONLY the latest user question in one short sentence. "
            "Do not repeat earlier sentences or explain grammar. If asked which number is greater, reply like: '10 is greater than 2.'"
        ),
    },
    {"role": "user", "content": "which is greater 2 or 10"},
    {"role": "assistant", "content": "10 is greater than 2."},
]

print("Local LLM Chatbot (TinyLlama) - type 'exit' or 'quit' to stop.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chat.")
        break
    response = get_response(pipe, history, user_input)
    print("AI:", response)