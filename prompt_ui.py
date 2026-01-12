from transformers.pipelines import pipeline
import streamlit as st

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

@st.cache_resource
def load_model():
    return pipeline(
        "text-generation",
        model=MODEL_NAME,
        max_new_tokens=100,
        device="cpu",
    )

def generate_response(pipe, user_input: str) -> str:
    messages = [{"role": "user", "content": user_input}]
    print("User input", user_input)
    input_text = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    result = pipe(input_text)
    generated_text = result[0].get("generated_text") or result[0].get("text", "")
    if "<|assistant|>" in generated_text:
        assistant_part = generated_text.split("<|assistant|>", 1)[1]
        response = assistant_part.split("<|", 1)[0].strip()
    else:
        response = generated_text[len(input_text):].strip()
    return response or "No response generated."

pipe = load_model()

st.header("Atul's Personal Chat Model")

response = None
with st.form(key="my_form"):
    user_input = st.text_input("Enter your prompt:")
    submitted = st.form_submit_button(label="Submit")
    if submitted and user_input:
        response = generate_response(pipe, user_input)
    elif submitted:
        response = "Please enter a prompt."

if response:
    st.text(response)
