from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import torch
import os
import time
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "nyayshayak")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Initializing Embeddings on: {device.upper()}")

embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': device}
)

pc = Pinecone(api_key=PINECONE_API_KEY)


if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,  
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=PINECONE_ENVIRONMENT
        )
    )
    print(f"Index '{PINECONE_INDEX_NAME}' created successfully")
else:
    print(f"Using existing Pinecone index: {PINECONE_INDEX_NAME}")


async def store_case_data(full_text: str, case_id: str):
    
    try:
        print(f"Storing case data for case_id: {case_id}")
        print(f"Full text length: {len(full_text)} characters")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2500,
            chunk_overlap=500,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        
        chunks = text_splitter.create_documents(
            texts=[full_text], 
            metadatas=[{"case_id": str(case_id), "collection": "case_files", "full_text_length": len(full_text)}]
        )
        
        print(f"Created {len(chunks)} chunks")

       
        vectorstore = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME,
            embedding=embedding_function
        )
        
       
        ids = vectorstore.add_documents(documents=chunks)
        
        print(f"Successfully stored {len(chunks)} chunks for Case {case_id}")
        print(f"Document IDs: {ids[:3]}..." if len(ids) > 3 else f"Document IDs: {ids}")
        
      
        print("Waiting for Pinecone to index documents...")
        time.sleep(2)  
        
       
        index = pc.Index(PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()
        print(f"Index stats - Total vectors: {stats.get('total_vector_count', 0)}")
        
        return f"Stored {len(chunks)} chunks for Case {case_id}"
    except Exception as e:
        print(f"Error storing case data: {str(e)}")
        import traceback
        traceback.print_exc()
        raise Exception(f"Failed to store case data for {case_id}: {str(e)}")


async def get_relevant_context(query: str, collection_name: str = "case_files", filter: dict = None, top_k: int = 5):
   
    try:
        print(f"Querying Pinecone for: query='{query[:50]}...', collection='{collection_name}', filter={filter}, top_k={top_k}")
        
       
        vectorstore = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME,
            embedding=embedding_function
        )
        
        pinecone_filter = {"collection": {"$eq": collection_name}}
        if filter:
            
            for key, value in filter.items():
                pinecone_filter[key] = {"$eq": str(value)}  
        
        print(f"Pinecone filter: {pinecone_filter}")  
        
       
        results = vectorstore.similarity_search(
            query=query,
            k=top_k,
            filter=pinecone_filter
        )
        
        print(f"Found {len(results)} matching documents")
        
        if results:
            print(f"First result metadata: {results[0].metadata}")
        else:
            print("No results found. Trying without filter...")
            results = vectorstore.similarity_search(
                query=query,
                k=top_k,
                filter={"collection": {"$eq": collection_name}}
            )
            print(f"Found {len(results)} documents in collection without case filter")
            
            if results:
                print(f"Available case_ids in results: {[r.metadata.get('case_id') for r in results]}")
        
        context_chunks = [doc.page_content for doc in results]
        
        return context_chunks
    except Exception as e:
        print(f"Error retrieving context: {str(e)}")
        import traceback
        traceback.print_exc()
        raise Exception(f"Failed to retrieve context for query '{query}': {str(e)}")