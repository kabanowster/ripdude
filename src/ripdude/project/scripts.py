import uvicorn


def ripdude():
    uvicorn.run("ripdude.server:app", host="0.0.0.0", port=8000, env_file="env")