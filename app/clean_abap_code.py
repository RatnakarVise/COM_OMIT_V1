import os
import gc
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import SystemMessage, HumanMessage
# from app.abap_explanation import extract_abap_explanation

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
 load_dotenv(dotenv_path=env_path)
else:
    print(f"Warning: .env file not found at {env_path}. Environment variables may not be set correctly.")
    
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

#Load RAG knowledge base
rag_file_path = os.path.join(os.path.dirname(__file__), "rag_knowledge_base.txt")
loader = TextLoader(file_path=rag_file_path, encoding="utf-8")
documents = loader.load()

# Create chunks for vector search
text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

embedding = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embedding)
retriever = vectorstore.as_retriever()


def clean_abap_code(code: str) -> str:
    # explanation = extract_abap_explanation(abap_code)
    # formatted_description = generate_description_from_explanation(explanation)

    retrieved_docs = retriever.get_relevant_documents(code)
    retrieved_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    if not retrieved_context.strip():
        return "No relevant context found in RAG knowledge base."

    # Final prompt for TSD generation
    prompt_template = ChatPromptTemplate.from_template(
            """
            You are an SAP ABAP code cleanup assistant.
            Using the following rules and examples, clean the ABAP code provided.

            Rules and Examples:
            {context}

            ABAP Code:
            {code}

            Cleaned ABAP Code:
            """
        )

    messages = prompt_template.format_messages(
        context=retrieved_context,
        code=code,
    )
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)
    response = llm.invoke(messages)
    return response.content if hasattr(response, "content") else str(response)
