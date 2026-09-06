from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import run_agent
from config import AGENT_NAME

app = FastAPI(title="Agent-Ripdude Execution Pod")

class TaskRequest(BaseModel):
    prompt: str

class TaskResponse(BaseModel):
    status: str
    agentId: str
    result: str

@app.post("/execute", response_model=TaskResponse)
def execute_task(request: TaskRequest):
    try:
        # Execute agent turn against local model and mounted files
        output = run_agent(
            userPrompt=request.prompt
        )
        return TaskResponse(
            status="success",
            agentId=AGENT_NAME,
            result=output
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/healthz")
def health():
    return {"status": "alive", "agent": AGENT_NAME}