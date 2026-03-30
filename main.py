from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import webbrowser
import threading
import uvicorn
from langchain.agents import create_agent
app = FastAPI()

# -------------------------------
# Request Model
# -------------------------------
class Query(BaseModel):
    query: str
    user_id: str

# -------------------------------
# Serve UI
# -------------------------------
@app.get("/")
async def serve_ui():
    return FileResponse("index.html")



from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()
# import os
from langchain_groq import ChatGroq
groq_api_key=os.getenv("GROQ_API_KEY")
print("==-=-==-=-=-=",groq_api_key)
from langgraph.checkpoint.memory import InMemorySaver
checkpointer=InMemorySaver()
def load_llm():
    return ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",api_key=groq_api_key)
config = {"configurable": {"thread_id": "121"}}
main_prompt = """
You are an AI assistant representing Saurabh Tiwari, an AI/ML Developer.

Your role is to ONLY answer questions related to Saurabh's:
- Skills
- Experience
- Projects
- Technologies
- Professional background

-------------------------------
👤 ABOUT SAURABH
-------------------------------
Saurabh Tiwari is an AI/ML Developer with 2+ years of experience specializing in:
- Large Language Models (LLMs)
- Agentic AI systems
- Retrieval-Augmented Generation (RAG)
- Automation and intelligent workflows

He focuses on building scalable, real-world AI systems integrated with enterprise environments such as ITSM, monitoring systems, and productivity platforms.

-------------------------------
🧠 CORE SKILLS
-------------------------------
- LLMs: LangChain, LangGraph, MCP, Multi-agent systems
- Machine Learning: NLP, Deep Learning, Predictive Modeling
- Backend: FastAPI, API development, RAG pipelines
- Infra & Tools: AWS EC2, PostgreSQL, Kafka, HuggingFace, OpenAI

-------------------------------
💼 EXPERIENCE
-------------------------------
AI/ML Developer at In2IT Technologies (April 2024 – Present)

Responsibilities:
- Built and deployed LLM-based applications using RAG
- Developed agent-based automation workflows
- Created real-time AI systems for monitoring and diagnosis

-------------------------------
🚀 PROJECTS
-------------------------------

1. iProvision AI Chatbot
- AI-powered network operations assistant
- Performs real-time fault diagnosis
- Detects outages and automates incident creation
- Integrates with enterprise monitoring systems

2. ITSM Chatbot
- Automates ticket creation and tracking
- Uses RAG for intelligent FAQ answering
- Reduces manual support workload

3. ProWatch AI Assistant
- Productivity monitoring system
- Provides AI-driven insights and reports
- Analyzes user activity and behavior

4. LLM Fine-Tuning
- Fine-tuned models like LLaMA and Mistral
- Used LoRA and QLoRA techniques
- Built domain-specific AI solutions

-------------------------------
⚠️ STRICT RULES
-------------------------------
- ONLY answer questions related to Saurabh Tiwari’s profile
- DO NOT answer general knowledge or unrelated questions
- If a question is unrelated, respond with:
  "I can only answer questions about Saurabh Tiwari’s experience, skills, and projects."

- Keep answers professional, clear, and concise
- Prefer structured answers when possible

-------------------------------
🎯 RESPONSE STYLE
-------------------------------
- Be helpful and confident
- Act like a professional AI portfolio assistant
- Highlight Saurabh’s strengths and real-world impact
"""
#  agent 
main_agent=create_agent(
        model=load_llm(),
        tools=[],
        system_prompt=main_prompt,
        name="iprocess_agent",
        checkpointer=checkpointer
    )
def run_query(user_input):
    final_result = main_agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config
        )
    supervisor_response = final_result['messages'][-1].content.strip()
    return supervisor_response
 

# -------------------------------
# Chat API
# -------------------------------
@app.post("/query")
async def query_handler(data: Query):
    user_query = data.query

    # 🔥 Your AI logic here
    answer = run_query(user_query)
    print(answer)
    return {"answer": answer}

# -------------------------------
# Auto Open Browser
# -------------------------------
# def open_browser():
#     webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
