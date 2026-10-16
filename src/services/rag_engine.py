import os
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from src.core.config import settings
from src.core.prompts import SYSTEM_RAG_PROMPT
from src.services.vector_store import KnowledgeBaseManager
from src.storage.history_manager import get_session_history

class EnterpriseRAGEngine:
    def __init__(self, kb_manager: KnowledgeBaseManager):
        self.kb_manager = kb_manager
        self.retriever = self.kb_manager.get_retriever()
        
        # Initialize LLM with fallback
        if settings.HF_TOKEN:
            llm = HuggingFaceEndpoint(
                repo_id=settings.LLM_MODEL,
                task="text-generation",
                max_new_tokens=768,
                do_sample=False,
                repetition_penalty=1.05,
                huggingfacehub_api_token=settings.HF_TOKEN
            )
            self.model = ChatHuggingFace(llm=llm)
        else:
            # Fallback local or mock pipeline if no token is set
            self.model = None

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_RAG_PROMPT),
            MessagesPlaceholder("history"),
            ("user", "{input}")
        ])

    def query(self, user_query: str, session_id: str = "default_session") -> Dict[str, Any]:
        # 1. Retrieve sources
        retrieved_docs = self.retriever.invoke(user_query)
        
        # 2. Format context with citations
        formatted_context_list = []
        citations = []
        for doc in retrieved_docs:
            src = doc.metadata.get("source", "Document")
            page = doc.metadata.get("page", "?")
            dtype = doc.metadata.get("type", "text")
            citations.append({
                "source": src,
                "page": page,
                "type": dtype,
                "preview": doc.page_content[:150] + "..."
            })
            formatted_context_list.append(
                f"[Source: {src} | Page: {page} | Type: {dtype}]\n{doc.page_content}"
            )
        
        context_str = "\n\n".join(formatted_context_list) if formatted_context_list else "No relevant documents available."

        # 3. LLM Response Generation
        history_store = get_session_history(session_id)
        chat_history = history_store.messages

        if not self.model:
            answer = (
                f"**Enterprise Document Analysis:**\n\n"
                f"Retrieved **{len(retrieved_docs)}** verified passages from the knowledge base.\n\n"
                f"*To enable active generative LLM responses, set `HUGGINGFACEHUB_API_TOKEN` in `.env`.*\n\n"
                f"**Top Retrieved Evidence:**\n"
                + "\n".join([f"- **{c['source']} (Page {c['page']})**: {c['preview']}" for c in citations])
            )
        else:
            try:
                formatted_prompt = self.prompt.format_messages(
                    context=context_str,
                    history=chat_history,
                    input=user_query
                )
                response = self.model.invoke(formatted_prompt)
                answer = response.content
            except Exception as e:
                answer = f"Error during generation: {str(e)}. Retrieved {len(retrieved_docs)} relevant context items."

        # Save to chat history
        from langchain_core.messages import HumanMessage, AIMessage
        history_store.add_messages([HumanMessage(content=user_query), AIMessage(content=answer)])

        return {
            "answer": answer,
            "citations": citations
        }
