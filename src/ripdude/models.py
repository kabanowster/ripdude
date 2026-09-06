from langchain_openai import ChatOpenAI

from ripdude.config import REASONING_MODEL_NAME

reasoning_model = ChatOpenAI(model=REASONING_MODEL_NAME)