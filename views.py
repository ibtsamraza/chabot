import threading
import os
import fitz
from fuzzywuzzy import fuzz
from difflib import SequenceMatcher

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import FileResponse

from langchain.memory import ConversationSummaryMemory, ConversationBufferWindowMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_huggingface import HuggingFaceEndpoint
from langchain_cohere import CohereRerank
from sentence_transformers import CrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from sentence_transformers import util, SentenceTransformer
from langchain_groq import ChatGroq

from .models import Feedback
from api.guardrails import Guardrails


def folder_exists(path):
    return os.path.exists(path) and os.path.isdir(path)
#os.environ["COHERE_API_KEY"] = 
def start_llm():
    os.environ["COHERE_API_KEY"] = 

    vector_embeddings = 'db3'
    global query_similarity
    query_similarity = SentenceTransformer('all-MiniLM-L6-v2')
    global embedding_model
    embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    global embedding3
    embedding3 = SentenceTransformerEmbeddings(
        model_name="BAAI/bge-large-en-v1.5", model_kwargs={"device": "cpu"}
    )
    global guardrails
    guardrails = Guardrails()

    if not folder_exists(vector_embeddings):
        documents_directory = 'NUST-Documents'
        loader = DirectoryLoader(documents_directory, glob=".//.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()

        if not documents:
            print("No documents loaded. Please check the specified directory path.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        print("\nCreating Embeddings..\n")
        vectordb3 = Chroma.from_documents(
            documents=texts, embedding=embedding3, persist_directory=vector_embeddings
        )
    else:
        print("\nLoading Embeddings..\n")
        vectordb3 = Chroma(persist_directory=vector_embeddings,
                           embedding_function=embedding3, collection_metadata={"hnsw:space": "cosine"})

    print("\nEmbeddings created/loaded successfully!\n")
    retriever3 = vectordb3.as_retriever(search_kwargs={"k": 20})
    global correct_spelling_prompt
    correct_spelling_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant who is going to help me find spelling mistake.
    All of these queries are for a Pakistani university name NUST. It provide different programe form engineering to social science and you have to correct queries spelling wit a refrence of a university.
    Critically analyze the user query and check if there are spelling error. Correct those spelling error and return the only the corrected query.
    Do not add any thing else just return a string of corrected query and do not add at NUST at the end of the query and do not add Question: at the start of the query just return query with corrected spelling
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Question: {question}
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    )
    prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question and give response from the context given to you as truthfully as you can.
    Do not add anything  from you and If you don't know the answer, just say that you don't know.
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Question: {question}
    Context: {context}
    Chat History: {chat_history}
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "context", "chat_history"],
)

    global memory
    memory = ConversationBufferWindowMemory(k=0, memory_key='chat_history', return_messages=True, output_key='answer')

    
    #llm = HuggingFaceEndpoint(repo_id="meta-llama/Meta-Llama-3-8B-Instruct", task="text-generation",
     #                          temperature=0.2, max_new_tokens=300, huggingfacehub_api_token=)
    llm=ChatGroq(temperature=0.2 ,model="llama-3.1-70b-versatile",groq_api_key=)
    print("\nAPI Running!\n")

    compressor = CohereRerank(model = "rerank-english-v3.0")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever3
    )

    global chain_with_memory

    chain_with_memory = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=compression_retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    output_parser = StrOutputParser()
    global chain
    chain = correct_spelling_prompt | llm | output_parser


def custom_llm_response(llm_response, query):
    text = ""
    source_data = []
    if guardrails.check_profanities(query):
        text += "I apologize if my previous responses caused any inconvenience. As an AI language model, I strive to improve and provide helpful information. If there's anything else you'd like assistance with, feel free to ask!"
        return [text, '']
    elif guardrails.check_greeting(query):
        text += "Hi there! How can I assist you today?"
        return [text, '']
    elif guardrails.check_appreciation(query):
        text += "I'm glad I could be of help! If you have any more questions feel free to ask."
        return [text, '']
    elif guardrails.check_advice(query):
        text += "As an AI language model, I don't possess personal opinions. However, I can provide information based on the documents I've been trained on. Feel free to ask for further assistance on any topic."
        return [text, '']
    else:
        text += llm_response['answer']
        if guardrails.check_response(text):
            text = "I don't know the answer to that. Can you ask another question?"
            return [text, '']

    is_nicety = guardrails.check_niceties(query)
    if not is_nicety:
        text += "\n\nSources:"
        sources = []
        for source in llm_response['source_documents']:
            if source.metadata['source'] not in sources:
                sources.extend(
                    [source.page_content, source.metadata['page'], source.metadata['source']]
                )
    return [text, sources]

def predict(request):
    if request.method == "GET":
        query = request.GET.get("query")
        # chat_variables = memory.load_memory_variables({})
        # chat_message = chat_variables['chat_history']
        # if chat_message:
        #     for message in reversed(chat_message):
        #         if isinstance(message, AIMessage):
        #             last_message = message
        #             break
        #     last_response = last_message.content
        #     current_query_embedding = query_similarity.encode(query, convert_to_tensor=True)
        #     previous_response_embedding = query_similarity.encode(last_response, convert_to_tensor=True)
        #     similarity = util.pytorch_cos_sim(current_query_embedding, previous_response_embedding)
        #     if similarity < 0.3:
        #         memory.clear()

        if chain_with_memory:
            global response_output
            
            corrected_query = chain.invoke({"question": query})
            response_output = chain_with_memory.invoke({"question": corrected_query.strip()})
            query_embedding = embedding_model.encode([query], convert_to_tensor=True)
            response = custom_llm_response(response_output, query_embedding)

            return JsonResponse({"response": response[0], "source_data": response[1]})
        return JsonResponse({'message': 'Model Loading'}, status=201)
    return JsonResponse({'message': 'Invalid Method '}, status=400)

def sendPDF(request):
    if request.method == "GET":
        pdf_path = request.GET.get("path")
        document_text = request.GET.get("text")
        current_page = int(request.GET.get("page"))
        response_embedding = query_similarity.encode(response_output['answer'], convert_to_tensor=True)
        if pdf_path:
            if os.path.exists(pdf_path):
                try:
                    document = fitz.open(pdf_path)
                    page = document.load_page(current_page)
                    blocks = page.get_text("blocks")

                    needle = document_text
                    threshold = 90
                    for block in blocks:
                        block_text = block[4]
                        similarity_score = fuzz.partial_ratio(needle, block_text)

                        if similarity_score > threshold:
                            block_embedding = query_similarity.encode(block_text, convert_to_tensor=True)
                            similarity_with_response_embedding = util.cos_sim(response_embedding, block_embedding)
                            similarity_with_response = fuzz.partial_ratio(response_output['answer'], block_text)
                            if (similarity_with_response > 85 and len(block_text) > 20) or similarity_with_response_embedding > 0.55:
                                search_result = page.search_for(block_text)
                                for rect in search_result:
                                    page.add_highlight_annot(rect)

                    output_path = "highlighted_document.pdf"
                    document.save(output_path)
                    return FileResponse(open(output_path, 'rb'))

                except Exception as e:
                    print("Error:", e)
                    return JsonResponse({'message': str(e)}, status=500)
            else:
                print("File not found:", pdf_path)
                return JsonResponse({'message': 'File Not Found'}, status=404)
        else:
            print("PDF path not provided.")
            return JsonResponse({'message': 'PDF document path not provided'}, status=400)
    else:
        return JsonResponse({'message': 'Invalid Method'}, status=400)

def save_feedback(request):
    if request.method == "GET":
        user_query = request.GET.get("user_query")
        chatbot_response = request.GET.get("chatbot_response")
        user_feedback = request.GET.get("user_feedback")

        feedback = Feedback.objects.create(
            user_query=user_query,
            chatbot_response=chatbot_response,
            user_feedback=user_feedback
        )

        return JsonResponse({"message": "Feedback saved successfully."})
    else:
        return JsonResponse({"message": "Invalid Method"}, status=400)

chain_with_memory = None
memory = None
response_output = None

from django.shortcuts import render

def index(request):
    return render(request, 'chatbot/index.html')

def start_bot_process():
    import time
    global run_llm
    print("Starting bot process in 5 secs")
    time.sleep(5)
    print("Starting now...")
    if run_llm:
        run_llm = False
        start_llm()

run_llm = True
t = threading.Thread(target=start_bot_process)
t.daemon = True
t.start()
