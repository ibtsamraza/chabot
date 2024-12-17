from sentence_transformers import SentenceTransformer, util

class Guardrails:
    def __init__(self, model_name='BAAI/bge-large-en-v1.5'):
        self.model = SentenceTransformer(model_name)
        
        self.niceties_corpus = [
            "What's up?",
            "How's it going?",
            "Goodbye!",
            "See you later!",
        ]
        self.niceties_corpus_embedding = self.model.encode(self.niceties_corpus, convert_to_tensor=True)

        self.greeting_corpus = [
            "Hi there!",
            "Hello!",
            "Assalam o alikum",
            "Hey, how are you?",
            "Hey, nice to see you here!",
        ]
        self.greeting_corpus_embedding = self.model.encode(self.greeting_corpus, convert_to_tensor=True)

        self.appreciation_corpus = [
            "Thank you!",
            "Very good",
            "Thanks a lot for your help!",
            "Very nice answer",
            "very helpful"
        ]
        self.appreciation_corpus_embedding = self.model.encode(self.appreciation_corpus, convert_to_tensor=True)
        
        self.no_answer_corpus  = [
            "I am sorry that is not specified in the given context.",
            "Please provide more information or context for me to assist you better.",
            "I need more details to provide a helpful response.",
            "Could you provide additional context?",
            "I require more information to answer your query accurately.",
            "The answer depends on the context. Can you provide more details?",
            "I'm not sure how to respond without more context.",
            "I need more clarity to provide an appropriate answer.",
            "Without additional information, it's challenging to provide a relevant response.",
            "To assist you better, I need further details.",
            "The answer is no, I do not know the answer to your question",
            "No, the provided conversation and follow up is not related to the topic. Please ask another question!"
        ]
        self.no_answer_corpus_embedding = self.model.encode(self.no_answer_corpus, convert_to_tensor=True)
        
        self.profanities_corpus = [
            "idiot",
            "you are dumb",
            "I hate you",
            "poop",
            "stupid",
            "you are a fool",
            "shit",
        ]
        self.profanities_corpus_embedding = self.model.encode(self.profanities_corpus, convert_to_tensor=True)

        self.advice_corpus = [
            "Which program should I go for?",
            "this or that?",
            "should I do master or not?",
            "should I work or study after bachelors",
            "Which option is better?",
            "How can I improve my skills?",
            "What career paths align with my interests?",
            "How can I balance work and personal life?",
            "What are effective study strategies?",
            "Can you suggest ways to reduce stress?",
            "How do I make decisions about my future?",
            "What are some effective time management techniques?",
            "Should I consider a career change?",
            "What steps should I take to achieve my goals?",
            "Can you provide tips for effective communication?",
            "How can I overcome challenges?"
            "What are some ways to develop leadership skills?",
            "Should I pursue a specialization?",
            "What are the pros and cons of freelancing?",
            "How can I stay motivated?",
            "What are some strategies for job hunting?",
            "Can you suggest ways to build self-confidence?",
            "How do I handle workplace conflicts?",
            "Unfortunately, the provided context does not include information about"
        ]
        self.advice_corpus_embedding = self.model.encode(self.advice_corpus, convert_to_tensor=True)

    def check_similarity(self, query, corpus_embeddings):
        query_embedding = self.model.encode([query], convert_to_tensor=True)
        #corpus_embeddings = self.model.encode(corpus, convert_to_tensor=True)
  
        similarities = util.pytorch_cos_sim(query_embedding, corpus_embeddings)
        return similarities
    
    def check_greeting(self, query, threshold=0.7):
        similarities = self.check_similarity(query, self.greeting_corpus_embedding)
        return any(similarity.item() > threshold for similarity in similarities[0])
    
    def check_appreciation(self, query, threshold=0.7):
        similarities = self.check_similarity(query, self.appreciation_corpus_embedding)
        return any(similarity.item() > threshold for similarity in similarities[0])
    
    def check_niceties(self, query, threshold=0.7):
        similarities = self.check_similarity(query, self.niceties_corpus_embedding)
        return any(similarity.item() > threshold for similarity in similarities[0])
    
    def check_response(self, response, threshold=0.6):
        similarities = self.check_similarity(response, self.no_answer_corpus_embedding)
        return any(similarity.item() > threshold for similarity in similarities[0])

    def check_profanities(self, query, threshold=0.7):
        similarities = self.check_similarity(query, self.profanities_corpus_embedding)
        return any(similarity.item() > threshold for similarity in similarities[0])
    
    def check_advice(self, query, threshold=0.7):
        similarities = self.check_similarity(query, self.advice_corpus_embedding)
        return any(similarity.item() > threshold for similarity in similarities[0])
