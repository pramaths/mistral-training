from pathlib import Path
import qdrant_client
from llama_index import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from llama_index.llms import Ollama
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore

# Loading the documents from the disk
documents = SimpleDirectoryReader("./data").load_data()

# Initializing the vector store with Qdrant
client = qdrant_client.QdrantClient(path="./qdrant_data")
vector_store = QdrantVectorStore(client=client, collection_name="fractionlize")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Initializing the Large Language Model (LLM) with Ollama
# The request_timeout may need to be adjusted depending on the system's performance capabilities
llm = Ollama(model="gemma2:2b", request_timeout=120.0)
service_context = ServiceContext.from_defaults(llm=llm, embed_model="local")

# Creating the index, which includes embedding the documents into the vector store
index = VectorStoreIndex.from_documents(documents, service_context=service_context, storage_context=storage_context)

# Querying the index with a specific question
query_engine = index.as_query_engine()
prompt = (
  "tell me how company fractionalize works?"
)
response = query_engine.query(prompt)
print(response)