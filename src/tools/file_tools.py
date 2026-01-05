"""
LangChain tools for file operations.
These tools wrap file_handler service for reading, writing, and discovering Python files.
"""

import os
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime

from src.services.file_handler import file_service


# Context schema for tools
@dataclass
class CodeContext:
    """Runtime context for code tools."""
    sandbox_dir: str
    target_dir: str


@tool
def read_file(file_path: str, runtime: ToolRuntime[CodeContext]) -> str:
    """Read the contents of a Python file.
    
    Use this tool to read and examine Python code files before analyzing or fixing them.
    
    Args:
        file_path: Path to the file (relative to target_dir or absolute path)
    
    Returns:
        File contents as string, or error message if file cannot be read
    """
    # Resolve path relative to target_dir if not absolute
    if not os.path.isabs(file_path):
        file_path = os.path.join(runtime.context.target_dir, file_path)
    
    content = file_service.read_file_to_text(file_path)
    if content is None:
        return f"Error: Could not read file {file_path}"
    return content


@tool
def write_file(file_path: str, content: str, runtime: ToolRuntime[CodeContext]) -> str:
    """Write content to a Python file. Only works within sandbox directory for security.
    
    Use this tool to write fixed or modified code back to files.
    IMPORTANT: This tool enforces security - it will only write files within the sandbox directory.
    
    Args:
        file_path: Path to the file (relative to target_dir or absolute path)
        content: Code content to write (complete file content)
    
    Returns:
        Success message or error message if write failed or security violation
    """
    # Resolve path
    if not os.path.isabs(file_path):
        file_path = os.path.join(runtime.context.target_dir, file_path)
    
    # Security check: ensure file is within sandbox
    abs_path = os.path.abspath(file_path)
    sandbox_abs = os.path.abspath(runtime.context.sandbox_dir)
    
    if not abs_path.startswith(sandbox_abs):
        return f"Error: Security violation - cannot write outside sandbox directory {sandbox_abs}. Attempted to write to: {abs_path}"
    
    success = file_service.write_text_to_file(file_path, content)
    if success:
        return f"Successfully wrote to {file_path}"
    else:
        return f"Error: Failed to write to {file_path}"


@tool
def list_python_files(directory: str, runtime: ToolRuntime[CodeContext]) -> str:
    """List all Python files in a directory recursively.
    
    Use this tool to discover all Python files that need to be analyzed or fixed.
    
    Args:
        directory: Directory to search (relative to target_dir or absolute path)
    
    Returns:
        Newline-separated list of Python file paths (relative to directory)
    """
    if not os.path.isabs(directory):
        directory = os.path.join(runtime.context.target_dir, directory)
    
    if not os.path.exists(directory):
        return f"Error: Directory {directory} does not exist"
    
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip __pycache__ and other common non-source directories
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'venv', 'node_modules']]
        
        for file in files:
            if file.endswith('.py'):
                rel_path = os.path.relpath(os.path.join(root, file), directory)
                python_files.append(rel_path)
    
    if python_files:
        return "\n".join(python_files)
    else:
        return "No Python files found in the directory"
