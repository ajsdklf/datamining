from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import os
import pandas as pd
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI as OpenAI_llama
from llama_index.core.llms import ChatMessage
from pathlib import Path
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.core.schema import TextNode
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core import PromptTemplate
from llama_index.core.schema import Document

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

client_llama = OpenAI_llama(model='gpt-4o-2024-05-13')

embed_model = OpenAIEmbedding(model="text-embedding-3-large", dimensions=128)
Settings.embed_model = embed_model

df_R = pd.read_csv('./datamining_data/롤스_원전.csv')
documents_R = [Document(text=row['텍스트']) for _, row in df_R.iterrows()]
index_R = VectorStoreIndex.from_documents(documents_R)
index_R.storage_context.persist("./datamining_data/롤스원전DB")

df_N = pd.read_csv('./datamining_data/노직_원전.csv')
documents_N = [Document(text=row['텍스트']) for _, row in df_N.iterrows()]
index_N = VectorStoreIndex.from_documents(documents_N)
index_N.storage_context.persist("./datamining_data/노직원전DB")

df_S = pd.read_csv('./datamining_data/싱어_원전.csv')
documents_S = [Document(text=row['텍스트']) for _, row in df_S.iterrows()]
index_S = VectorStoreIndex.from_documents(documents_S)
index_S.storage_context.persist("./datamining_data/싱어원전DB")

df_K = pd.read_csv('./datamining_data/칸트_원전.csv')
documents_K = [Document(text=row['텍스트']) for _, row in df_K.iterrows()]
index_K = VectorStoreIndex.from_documents(documents_K)
index_K.storage_context.persist("./datamining_data/칸트원전DB")

df_H = pd.read_csv('./datamining_data/홉스_원전.csv')
documents_H = [Document(text=row['텍스트']) for _, row in df_H.iterrows()]
index_H = VectorStoreIndex.from_documents(documents_H)
index_H.storage_context.persist("./datamining_data/홉스원전DB")
