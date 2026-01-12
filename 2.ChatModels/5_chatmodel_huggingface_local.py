from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=2.0,
        max_new_tokens=512,
    )
)
model = ChatHuggingFace(llm=llm)
result = model.invoke("who is lunatic coder? share it's youtube channel link. make sure to check with latest data of 2026")
print(result.content)