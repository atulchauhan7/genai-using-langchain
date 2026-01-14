from transformers.pipelines import pipeline
import streamlit as st

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

@st.cache_resource
def load_model():
    return pipeline(
        "text-generation",
        model=MODEL_NAME,
        # max_new_tokens=100,
        device="cpu",
    )

def generate_response(pipe, user_input: str, gen_params: dict) -> str:
    messages = [{"role": "user", "content": user_input}]
    input_text = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    result = pipe(input_text, **gen_params)
    generated_text = result[0].get("generated_text") or result[0].get("text", "")
    if "<|assistant|>" in generated_text:
        assistant_part = generated_text.split("<|assistant|>", 1)[1]
        response = assistant_part.split("<|", 1)[0].strip()
    else:
        response = generated_text[len(input_text):].strip()
    return response or "No response generated."

pipe = load_model()

st.header("Atul's Personal Chat Model")

with st.sidebar:
    st.subheader("Generation Settings")
    max_new_tokens = st.slider("Response length (tokens)", 64, 1024, 256, step=16)
    min_new_tokens = st.slider("Min new tokens", 0, 512, 64, step=16)
    do_sample = st.checkbox("Enable sampling", value=True)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, step=0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.95, step=0.05)
    repetition_penalty = st.slider("Repetition penalty", 1.0, 2.0, 1.1, step=0.05)

    if min_new_tokens > max_new_tokens:
        min_new_tokens = max_new_tokens

response = None
with st.form(key="my_form"):
    user_input = st.text_input("Enter your prompt:")
    submitted = st.form_submit_button(label="Submit")
    if submitted and user_input:
        gen_params = {
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
        }
        with st.spinner("Generating response..."):
            response = generate_response(pipe, user_input, gen_params)
    elif submitted:
        response = "Please enter a prompt."

if response:
    st.text(response)
