import os

HARNESS_ROOT = os.getenv("HARNESS_ROOT")
if not HARNESS_ROOT:
    raise ValueError("HARNESS_ROOT environment variable is not set.")

WORKSPACE_DIR = os.getenv("WORKSPACE_DIR")
if not WORKSPACE_DIR:
    raise ValueError("WORKSPACE_DIR environment variable is not set.")

AGENT_NAME = os.getenv("AGENT_NAME")
if not AGENT_NAME:
    raise ValueError("AGENT_NAME environment variable is not set.")

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
if not OPENAI_BASE_URL:
    raise ValueError("OPENAI_BASE_URL environment variable is not set.")

MODEL_NAME = os.getenv("MODEL_NAME")
if not MODEL_NAME:
    raise ValueError("MODEL_NAME environment variable is not set.")