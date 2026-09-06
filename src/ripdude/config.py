import os
from logging import warning

#
# Mandatories
#

REASONING_MODEL_NAME = os.getenv("REASONING_MODEL_NAME")
if not REASONING_MODEL_NAME:
    raise ValueError("REASONING_MODEL_NAME environment variable is not set.")

AGENT_NAME = os.getenv("AGENT_NAME")
if not AGENT_NAME:
    raise ValueError("Required AGENT_NAME environment variable is not set.")

OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
if not OPENAI_API_BASE:
    raise ValueError("OPENAI_API_BASE environment variable is not set.")

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

#
# Having defaults in dockerfile
#
SYSTEMONE_URL = os.getenv("SYSTEMONE_URL")
# if not SYSTEMONE_URL:
#     raise ValueError(f"Required SYSTEMONE_URL environment variable is not set.")

# /harness
HARNESS_ROOT = os.getenv("HARNESS_ROOT")
if not HARNESS_ROOT:
    raise ValueError(f"Required HARNESS_ROOT environment variable is not set.")

# /workspace
WORKSPACE_ROOT = os.getenv("WORKSPACE_ROOT")
if not WORKSPACE_ROOT:
    raise ValueError(f"Required WORKSPACE_ROOT environment variable is not set.")

# WORKER_MODEL_NAME = os.getenv("WORKER_MODEL_NAME", "worker")
# if not WORKER_MODEL_NAME:
#     raise ValueError("WORKER_MODEL_NAME environment variable is not set.")

# redis://redis:6379/0
REDIS_URL = os.getenv("REDIS_URL")
if not REDIS_URL:
    raise ValueError("REDIS_URL environment variable is not set.")

# http://cognee-mcp:8000/mcp
COGNEE_MCP_URL = os.getenv("COGNEE_MCP_URL")
if not COGNEE_MCP_URL:
    raise ValueError("COGNEE_MCP_URL environment variable is not set.")

# http://playwright:3080/mcp
PLAYWRIGHT_MCP_URL = os.getenv("PLAYWRIGHT_MCP_URL")
if not PLAYWRIGHT_MCP_URL:
    raise ValueError("PLAYWRIGHT_MCP_URL environment variable is not set.")

REDIS_TTL_DAYS = int(os.getenv("REDIS_TTL_DAYS", 7))

#
# Optional
#

MCP_TOOLS = os.getenv("MCP_TOOLS")
if not MCP_TOOLS:
    warning(f"\nNo additional MCP Tools are set for the agent {AGENT_NAME}.\n")

AUTOLOAD_MODEL = os.getenv("AUTOLOAD_MODEL", "false").lower() in ["true", "yes", "1", "on"]