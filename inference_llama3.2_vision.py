from langchain_core.prompt_values import PromptValue
from huggingface_hub import InferenceClient
from langchain_core.messages import (
    MessageLikeRepresentation,
)
import PIL
from PIL import Image
import base64
from io import StringIO, BytesIO

def _getHuggingFaceClient():
    client = InferenceClient(
    model="meta-llama/Llama-3.2-11B-Vision-Instruct",
    token="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"
    )
    return client


def lvlm_inference_with_conversation(conversation, max_tokens: int = 1000, temperature: float = 0.95, top_p: float = 0.1, top_k: int = 10):
    # get PredictionGuard client
    client = _getHuggingFaceClient()
    # get message from conversation
    #messages = conversation.get_message()
    # call chat completion endpoint at Grediction Guard
    response = client.chat_completion(
        model="meta-llama/Llama-3.2-11B-Vision-Instruct",
        messages=conversation,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        
    )
    return response['choices'][-1]['message']['content']

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# class Conversation:
#     def __init__(self, system=""):
#         self.messages = []
#         if system:
#             self.messages.append({"role": "system", "content": system})
#     def generate(self, user_question, model=8, temp=0):
#         self.messages.append({"role": "user", "content":user_question})
#         response = llama31(self.messages, model, temperature=temp)
#         self.messages.append({"role":"assistant", "content":response})
#         return response
b64_image = encode_image('./output_images/page_5.png')
data_uri = f"data:image/png;base64,{b64_image}"
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image_url","image_url": {"url": data_uri}},
            {"type": "text", "text": "i will provide you with an image which contain multiple figures and you have to analyze the image and extract all the data present in it: "}
        ]
    }
]
result_llama32 = lvlm_inference_with_conversation(messages)
print(result_llama32)