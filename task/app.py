import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.vectorstores import VectorStore
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY


SYSTEM_PROMPT = """You are a RAG-powered assistant that assists users with their questions about microwave usage.
            
## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
"""

USER_PROMPT = """##RAG CONTEXT:
{context}


##USER QUESTION: 
{query}"""


class MicrowaveRAG:
    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = self._setup_vectorstore()

    def _setup_vectorstore(self) -> VectorStore:
        """Initialize or load the vector store."""
        print("Initializing Microwave Manual RAG System...")
        index_path = os.path.join(os.path.dirname(__file__), "microwave_faiss_index")

        if os.path.exists(index_path):
            print("Existing FAISS index found. Loading from disk...")
            return FAISS.load_local(
                folder_path=index_path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
            )

        print("No existing index found. Creating a new one...")
        return self._create_new_index()

    def _create_new_index(self) -> VectorStore:
        print("Loading text document...")

        manual_path = os.path.join(os.path.dirname(__file__), "microwave_manual.txt")
        loader = TextLoader(file_path=manual_path, encoding="utf-8")
        documents = loader.load()

        # Cut the manual into overlapping chunks so embeddings capture enough context.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", "."],
        )
        chunks = splitter.split_documents(documents)

        # Build and persist the FAISS index so we reuse embeddings on the next run.
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        index_path = os.path.join(os.path.dirname(__file__), "microwave_faiss_index")
        vectorstore.save_local(index_path)
        print("New FAISS index created and saved.")

        return vectorstore

    def retrieve_context(self, query: str, k: int = 4, score=0.3) -> str:
        """
        Retrieve the context for a given query.
        Args:
              query (str): The query to retrieve the context for.
              k (int): The number of relevant documents(chunks) to retrieve.
              score (float): The similarity score between documents and query. Range 0.0 to 1.0.
        """
        print(f"{'=' * 100}\nSTEP 1: RETRIEVAL\n{'-' * 100}")
        print(f"Query: '{query}'")
        print(f"Searching for top {k} most relevant chunks with similarity score {score}:")

        # FAISS returns the most similar chunks with a relevance score filter.
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query=query,
            k=k,
            score_threshold=score,
        )

        if not results:
            print("No relevant context found.")
            print("=" * 100)
            return ""

        context_parts = []
        for doc, similarity in results:
            context_parts.append(doc.page_content)
            print(f"Score: {similarity:.3f}")
            print(doc.page_content)
            print("-" * 30)

        print("=" * 100)
        return "\n\n".join(context_parts)  # join all chunks in one string with `\n\n` between chunks

    def augment_prompt(self, query: str, context: str) -> str:
        print(f"\nSTEP 2: AUGMENTATION\n{'-' * 100}")

        # Plug retrieved context and user query into the RAG prompt template.
        augmented_prompt = USER_PROMPT.format(context=context, query=query)

        print(f"{augmented_prompt}\n{'=' * 100}")
        return augmented_prompt

    def generate_answer(self, augmented_prompt: str) -> str:
        print(f"\nSTEP 3: GENERATION\n{'-' * 100}")

        # System prompt = behavior guardrails, Human message = context + user question.
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=augmented_prompt),
        ]
        response = self.llm_client.invoke(messages)
        print(response.content)
        return response.content


def main(rag: MicrowaveRAG):
    print("Microwave RAG Assistant")

    while True:
        user_question = input("\n> ").strip()
        if not user_question:
            continue
        if user_question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        context = rag.retrieve_context(user_question)
        augmented_prompt = rag.augment_prompt(user_question, context)
        rag.generate_answer(augmented_prompt)


main(
    MicrowaveRAG(
        embeddings=AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-3-small-1",
            azure_endpoint=DIAL_URL,
            api_key=SecretStr(API_KEY),
        ),
        llm_client=AzureChatOpenAI(
            temperature=0.0,
            azure_deployment="gpt-4o",
            azure_endpoint=DIAL_URL,
            api_key=SecretStr(API_KEY),
            api_version="",
        ),
    )
)
