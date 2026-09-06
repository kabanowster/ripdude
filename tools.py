import re
import datetime
import subprocess
from pathlib import Path
from langchain_core.tools import tool

from config import HARNESS_ROOT, AGENT_NAME, WORKSPACE_DIR


@tool
def exec_bash(command: str) -> str:
    """Execute a system bash command within the scope of the workspace directory (e.g., ls, grep, find, cat).
    Use this to search, filter, or inspect files. Use write_file for creating or editing code files.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKSPACE_DIR,
            capture_output=True,
            text=True,
            timeout=30
        )
        output = result.stdout if result.returncode == 0 else result.stderr
        return output if output.strip() else "Command executed with no output."
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds."
    except Exception as e:
        return f"Error executing command: {str(e)}"

# ------------------------------------------------------------------
# Dynamic Skill & File Tools
# ------------------------------------------------------------------
@tool
def get_skill(skillName: str) -> str:
    """Retrieves the full Standard Operating Procedure (.skill file) for a given skill name."""
    # Sanitizes input to prevent path traversal
    cleanName = Path(skillName).stem
    skillPath = Path(HARNESS_ROOT) / AGENT_NAME / "skills" / f"{cleanName}.skill"

    if skillPath.exists():
        return f"SOP for '{cleanName}':\n{skillPath.read_text(encoding='utf-8')}"
    return f"Error: Skill file '{cleanName}.skill' not found in skills directory."


@tool
def read_local_file(filePath: str) -> str:
    """Reads and returns content from a local file path."""
    p = Path(filePath)
    return p.read_text(encoding="utf-8") if p.exists() else f"Error: File '{filePath}' not found."


@tool
def write_local_file(filePath: str, content: str) -> str:
    """Writes content to a specified local file path."""
    p = Path(filePath)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Successfully updated {filePath}"


def trim_short_memory(clusterRoot: str, agentName: str, maxEntries: int = 50, daysLimit: int = 90):
    """Parses short-memory.md, filters entries older than daysLimit, and keeps at most maxEntries."""
    memoryPath = Path(clusterRoot) / agentName / "memories" / "short-memory.md"
    if not memoryPath.exists():
        return

    lines = memoryPath.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return

    # Calculate cutoff date (3 months ~ 90 days)
    cutoffDate = datetime.datetime.now() - datetime.timedelta(days=daysLimit)
    datePattern = re.compile(r'\[(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2})\]')

    validEntries = []

    for line in lines:
        if not line.strip():
            continue

        match = datePattern.search(line)
        if match:
            try:
                entryDate = datetime.datetime.strptime(match.group(1), "%Y-%m-%d %H:%M")
                # Retain entries newer than or equal to 3 months ago
                if entryDate >= cutoffDate:
                    validEntries.append(line)
            except ValueError:
                validEntries.append(line)  # Keep line if date formatting fails
        else:
            validEntries.append(line)  # Retain non-timestamped metadata lines

    # Keep only the most recent 'maxEntries' (top 50)
    trimmedEntries = validEntries[-maxEntries:]

    # Overwrite file with sanitized list
    memoryPath.write_text("\n".join(trimmedEntries) + "\n", encoding="utf-8")


@tool
def append_to_short_memory(clusterRoot: str, agentName: str, prompt: str, actionsTaken: list, finalResponse: str):
    """Appends an execution record to the agent short-term memory."""
    memoryPath = Path(clusterRoot) / agentName / "memories" / "short-memory.md"
    memoryPath.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    toolsSummary = f" (Tools: {', '.join(actionsTaken)})" if actionsTaken else " (No tools used)"

    cleanSummary = finalResponse.replace("\n", " ").strip()
    if len(cleanSummary) > 150:
        cleanSummary = cleanSummary[:150] + "..."

    logEntry = f"\n- [{timestamp}] Task: \"{prompt}\"{toolsSummary} -> Result: {cleanSummary}"

    with open(memoryPath, "a", encoding="utf-8") as f:
        f.write(logEntry)

    # Automatically prune entries older than 90 days or exceeding 50 items
    trim_short_memory(clusterRoot, agentName, maxEntries=50, daysLimit=90)


# ------------------------------------------------------------------
# Long-Memory Management Tools
# ------------------------------------------------------------------
@tool
def read_long_memory() -> str:
    """Reads all long-term memory entries with their line numbers.
    Always call this FIRST before updating or deleting an existing memory entry."""
    memoryPath = Path(HARNESS_ROOT) / AGENT_NAME / "memories" / "long-memory.md"
    if not memoryPath.exists():
        return "Long-memory file does not exist yet."

    lines = [line.strip() for line in memoryPath.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return "Long-memory file is currently empty."

    return "\n".join([f"{idx + 1}: {line}" for idx, line in enumerate(lines)])


@tool
def append_long_memory(journalNote: str) -> str:
    """Appends a new curated rule, preference, or architectural decision to long-memory.md."""
    memoryPath = Path(HARNESS_ROOT) / AGENT_NAME / "memories" / "long-memory.md"
    memoryPath.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    formattedNote = f"- [{timestamp}] {journalNote.strip()}\n"

    with open(memoryPath, "a", encoding="utf-8") as f:
        f.write(formattedNote)

    return f"Added new entry: '{journalNote}'"


@tool
def update_long_memory(entryNumber: int, updatedNote: str) -> str:
    """Replaces a specific long-memory entry (1-indexed line number) with updated text.
    Call `read_long_memory` first to inspect line numbers."""
    memoryPath = Path(HARNESS_ROOT) / AGENT_NAME / "memories" / "long-memory.md"
    if not memoryPath.exists():
        return "Error: long-memory.md does not exist."

    lines = [line.strip() for line in memoryPath.read_text(encoding="utf-8").splitlines() if line.strip()]

    if entryNumber < 1 or entryNumber > len(lines):
        return f"Error: Entry number {entryNumber} is out of bounds (total entries: {len(lines)})."

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    lines[entryNumber - 1] = f"- [{timestamp}] (Updated) {updatedNote.strip()}"

    memoryPath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return f"Successfully updated entry #{entryNumber}."


@tool
def delete_long_memory(entryNumber: int) -> str:
    """Deletes a specific long-memory entry (1-indexed line number).
    Call `read_long_memory` first to inspect line numbers."""
    memoryPath = Path(HARNESS_ROOT) / AGENT_NAME / "memories" / "long-memory.md"
    if not memoryPath.exists():
        return "Error: long-memory.md does not exist."

    lines = [line.strip() for line in memoryPath.read_text(encoding="utf-8").splitlines() if line.strip()]

    if entryNumber < 1 or entryNumber > len(lines):
        return f"Error: Entry number {entryNumber} is out of bounds (total entries: {len(lines)})."

    removedEntry = lines.pop(entryNumber - 1)
    memoryPath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return f"Successfully deleted entry #{entryNumber}: '{removedEntry}'"


tools = [
    get_skill,
    read_local_file,
    write_local_file,
    read_long_memory,
    append_long_memory,
    update_long_memory,
    delete_long_memory,
]
toolsByName = {t.name: t for t in tools}