"""RAG chain construction with LCEL for DocMind AI."""

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic


def get_llm(
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    temperature: float = 0,
) -> BaseChatModel:
    """Get a chat model instance.

    Args:
        provider: LLM provider ("openai" or "anthropic").
        model: Model name.
        temperature: Sampling temperature.

    Returns:
        A chat model instance.
    """
    if provider == "anthropic":
        return ChatAnthropic(model=model, temperature=temperature)
    # Default: OpenAI
    return ChatOpenAI(model=model, temperature=temperature)


def _format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into a context string."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        source_info = f"{source}" + (f" (page {page})" if page else "")
        formatted.append(f"[Source {i}: {source_info}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


# Prompt for contextualizing follow-up questions
_CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Given the chat history and the latest user question, "
        "reformulate the question to be a standalone question that "
        "doesn't require the chat history to understand. "
        "Do NOT answer the question — only reformulate it if needed, "
        "or return it as-is if it's already standalone.",
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Main RAG prompt
_RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are DocMind AI, a helpful assistant that answers questions "
        "based on the provided document context. Follow these rules:\n"
        "1. Answer ONLY based on the provided context.\n"
        "2. If the context doesn't contain enough information, say so clearly.\n"
        "3. Cite which source(s) you used (e.g., [Source 1], [Source 2]).\n"
        "4. Be concise but thorough.\n\n"
        "Context:\n{context}",
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


def create_rag_chain(
    retriever: BaseRetriever,
    llm: BaseChatModel,
):
    """Create a conversational RAG chain with LCEL.

    Args:
        retriever: The document retriever.
        llm: The chat model.

    Returns:
        A runnable RAG chain.
    """
    # Contextualize chain: rewrites follow-up questions to standalone
    contextualize_chain = _CONTEXTUALIZE_PROMPT | llm | StrOutputParser()

    def contextualize_question(input_dict: dict) -> str:
        """Rewrite question if there's chat history, otherwise pass through."""
        if input_dict.get("chat_history"):
            return contextualize_chain.invoke(input_dict)
        return input_dict["input"]

    # Full RAG chain
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: _format_docs(
                retriever.invoke(contextualize_question(x))
            )
        )
        | _RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain


def query(
    chain,
    question: str,
    chat_history: list[BaseMessage] | None = None,
) -> str:
    """Run a query through the RAG chain.

    Args:
        chain: The RAG chain.
        question: User question.
        chat_history: Previous conversation messages.

    Returns:
        The answer string.
    """
    return chain.invoke({
        "input": question,
        "chat_history": chat_history or [],
    })
