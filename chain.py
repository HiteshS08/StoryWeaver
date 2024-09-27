import os
import logging
from dotenv import load_dotenv
from datasets import load_dataset
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()
logging.basicConfig(level=logging.INFO)


class Chain:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(
            model_name='llama-3.1-8b-instant',
            groq_api_key=self.groq_api_key,
            temperature=0.7,
            max_tokens=4000
        )
        self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
        self.ner_model = AutoModelForTokenClassification.from_pretrained(
            "dbmdz/bert-large-cased-finetuned-conll03-english")
        self.nlp = pipeline("ner", model=self.ner_model, tokenizer=self.tokenizer, aggregation_strategy="simple")
        self.embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        self.vectorstore = self.load_vectorstore()
        self.prompt_template = PromptTemplate(
            input_variables=["content_premise", "genre", "key_points", "context"],
            template="""
You are a skilled writer.

Based on the following details, write a detailed and engaging piece:

Content Premise: {content_premise}
Genre/Style: {genre}
Key Points: {key_points}
Relevant Context: {context}

Ensure the piece is cohesive, follows the conventions of the {genre} genre or style, and incorporates the provided key points and context appropriately. Use clear language, vivid descriptions, and effectively develop the content to engage the reader.

### WRITTEN CONTENT (NO PREAMBLE):
"""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def load_vectorstore(self):
        persist_directory = 'db/'
        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embedding,
            )
            logging.info("Loaded existing ChromaDB vector store.")
        else:
            logging.info("Creating ChromaDB vector store from the dataset.")
            dataset = load_dataset('wikitext', 'wikitext-103-v1')
            texts = dataset['train']['text']
            docs = [Document(page_content=text) for text in texts]
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            split_texts = text_splitter.split_documents(docs)
            vectorstore = Chroma.from_documents(
                split_texts,
                embedding=self.embedding,
                persist_directory=persist_directory
            )
            vectorstore.persist()
            logging.info("Created and persisted ChromaDB vector store.")

        return vectorstore

    def extract_entities(self, prompt):
        ner_results = self.nlp(prompt)
        genre = ""
        key_points = []
        context = []

        for entity in ner_results:
            if entity['entity_group'] == 'MISC' and not genre:
                genre = entity['word']
            elif entity['entity_group'] == 'PER':
                key_points.append(entity['word'])
            elif entity['entity_group'] == 'LOC':
                context.append(entity['word'])

        if not genre:
            genre = 'General'

        return genre, key_points, context

    def generate_chapter(self, user_prompt):
        genre, key_points, context = self.extract_entities(user_prompt)
        content_premise = user_prompt
        key_points_str = ', '.join(key_points) if key_points else 'N/A'
        context_str = ', '.join(context) if context else 'N/A'
        chain_inputs = {
            'content_premise': content_premise,
            'genre': genre,
            'key_points': key_points_str,
            'context': context_str
        }

        try:
            res = self.chain.run(chain_inputs)
            logging.info("Successfully generated the content.")
            return res, genre, key_points_str, context_str
        except Exception as e:
            logging.error(f"Error generating the content: {e}")
            raise
