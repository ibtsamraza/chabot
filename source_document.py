

import io
from PIL import Image
from langchain_core.output_parsers import StrOutputParser
import requests
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from sentence_transformers import util, SentenceTransformer
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.memory import ConversationSummaryMemory, ConversationBufferWindowMemory



prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> you are an intelligent system ho is going to write a haiku for this {image}, it would be: "
    <|start_header_id|>user<|end_header_id|>
    text: {text}
    image: {image}
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["text", "image"],
)

llm=HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.2-11B-Vision-Instruct", task="text-generation",
                        temperature=0.1, huggingfacehub_api_token="hf_")
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
image = Image.open(requests.get(url, stream=True).raw)

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "If I had to write a haiku for this one, it would be: "}
    ]}
]
output_parser = StrOutputParser()

chain = prompt | llm | output_parser
print(chain.invoke({"text":"If I had to write a haiku for this one, it would be:", "image":image}))
