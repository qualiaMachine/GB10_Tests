# #calling fastapi to set up server for frontend
# from fastapi import FastAPI
# from pydantic import BaseModel
# import subprocess
# import json

# app = FastAPI()

# class Query(BaseModel):
#     input: str

# #making the file QWEN_multi_agent_pipeline-ver2.py callable (runs the script) to outside
# @app.post("/chat")
# def chat(query: Query):
#     cmd = [
#         "python",
#         "/home/ec2-user/SageMaker/QWEN_multi_agent_pipeline-ver2.py",
#         "--input",
#         query.input
#     ]
#     result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)

#     return {"response": result.stdout}
from fastapi import FastAPI
from pydantic import BaseModel
import subprocess

app = FastAPI()

class Query(BaseModel):
    input: str

@app.post("/chat")
def chat(query: Query):
    cmd = [
        "python",
        "-u",
        "/home/ec2-user/SageMaker/QWEN_multi_agent_pipeline-ver2.py",
        "--input",
        query.input
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Combine stdout + stderr to catch everything
    output = result.stdout.strip() or result.stderr.strip()

    return {"response": output}
