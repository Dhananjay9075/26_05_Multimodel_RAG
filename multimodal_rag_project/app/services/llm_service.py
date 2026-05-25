# title="app/services/llm_service.py"
import requests
from typing import List, Dict, Any
from app.config import settings

class LLMOnyxOrchestrator:
    @staticmethod
    def generate_response(query: str, vector_context: List[Dict[str, Any]], web_context: List[Dict[str, Any]], history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Fuses knowledge context vectors, synthesizes real-time updates, and builds clear, cited responses.
        """
        # 1. Format context payloads with explicit citation boundaries
        formatted_vector = ""
        for i, doc in enumerate(vector_context):
            formatted_vector += f"[Doc Citation {i+1}]: Source={doc['source']}, Page={doc['page']}\nContext:\n{doc['content']}\n\n"
            
        formatted_web = ""
        for i, web in enumerate(web_context):
            formatted_web += f"[Web Citation {i+1}]: Link={web['url']}, Title={web['title']}\nContext:\n{web['content']}\n\n"

        # 2. Construct clear system operational boundaries
        system_prompt = (
            "You are an Advanced Multimodal RAG Reasoning Agent.\n"
            "Your task is to answer user queries with absolute structural accuracy using the provided contexts.\n"
            "Always cite references inline using standard notation like [Doc Citation X] or [Web Citation Y].\n"
            "If the user asks to generate a document profile (e.g., PDF or DOCX file output), append a trailing structure block "
            "exactly formatted as: '[[FILE_GENERATION_REQUEST: TYPE=PDF_OR_DOCX_HERE]]' containing the final Markdown copy to export."
        )

        user_message = f"User Request: {query}\n\n"
        if formatted_vector:
            user_message += f"--- START UPLOADED DOCUMENT KNOWLEDGE VECTORS ---\n{formatted_vector}--- END ---\n\n"
        if formatted_web:
            user_message += f"--- START REAL-TIME WEB CONTEXTS ---\n{formatted_web}--- END ---\n\n"

        # 3. Compile history pipelines
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            for interaction in history:
                messages.append({"role": interaction["role"], "content": interaction["content"]})
        messages.append({"role": "user", "content": user_message})

        # 4. Route payload request to Groq Engine API
        if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "mock-groq-key":
            # High-fidelity mock generation framework for sandboxed local executions
            mock_text = f"Synthesized Response to: '{query}'. Based on internal vector knowledge and real-time live search contexts, operations are performing nominally. [Doc Citation 1]. \n\nIf you requested a document format download, it has been initialized."
            if "pdf" in query.lower() or "generate file" in query.lower():
                mock_text += "\n\n[[FILE_GENERATION_REQUEST: TYPE=PDF]]\n# Document Export\nGenerated intelligence output file matching parameter requirements successfully."
            elif "docx" in query.lower() or "word" in query.lower():
                mock_text += "\n\n[[FILE_GENERATION_REQUEST: TYPE=DOCX]]\n# Document Export\nGenerated intelligence output file matching parameter requirements successfully."
            return {"answer": mock_text, "citations": ["Doc Citation 1", "Web Citation 1"]}

        try:
            headers = {
                "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama-3.1-8b-instant",
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 4096
            }
            res = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=30)
            if res.status_code == 200:
                choice = res.json()["choices"][0]["message"]
                return {"answer": choice["content"], "citations": []}
            return { "answer": f"Groq Error: {res.text}", "citations": [] }
        except Exception as e:
            return {"answer": f"Inference engine exceptions occurred: {str(e)}", "citations": []}