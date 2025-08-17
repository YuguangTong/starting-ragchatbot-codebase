# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Quick start (recommended)
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Package Management
```bash
# Install dependencies
uv sync

# Add new dependency
uv add package_name

# Development dependencies
uv add --dev package_name
```

### Environment Setup
```bash
# Required environment file
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

## Architecture Overview

This is a RAG (Retrieval-Augmented Generation) system with a modular backend architecture orchestrated by a central `RAGSystem` class. The system uses Claude AI with function calling to autonomously search course materials stored in ChromaDB.

### Core Data Flow
1. **Document Ingestion**: Course documents → `DocumentProcessor` → chunks → `VectorStore` → ChromaDB
2. **Query Processing**: User query → `RAGSystem` → `AIGenerator` → Claude API (with tools) → response
3. **Tool-Based Search**: Claude decides to search → `CourseSearchTool` → `VectorStore` → semantic results → formatted response

### Key Architectural Components

**RAGSystem** (`rag_system.py`): Central orchestrator that coordinates all components and manages the query lifecycle. Handles session management, tool integration, and response generation.

**AIGenerator** (`ai_generator.py`): Manages Claude API interactions with function calling capabilities. Handles tool execution flow where Claude can autonomously decide to search for information.

**VectorStore** (`vector_store.py`): ChromaDB wrapper with dual collections - one for course metadata and one for content chunks. Provides unified search interface with course/lesson filtering.

**CourseSearchTool** (`search_tools.py`): Function calling tool that enables Claude to search course content semantically. Implements the Tool interface and tracks search sources for UI display.

**DocumentProcessor** (`document_processor.py`): Converts raw course documents into structured `Course` objects and searchable `CourseChunk` objects with metadata.

**SessionManager** (`session_manager.py`): Maintains conversation history per session for context-aware responses across multiple queries.

### Data Models
- **Course**: Represents a complete course with title, instructor, and lessons
- **Lesson**: Individual lesson with number, title, and optional link
- **CourseChunk**: Text chunk with course/lesson metadata for vector storage

### Configuration
All settings are centralized in `config.py` using a dataclass pattern:
- Anthropic API settings (model: claude-sonnet-4-20250514)
- Embedding model (all-MiniLM-L6-v2)
- Document processing parameters (chunk size: 800, overlap: 100)
- ChromaDB storage path

### Frontend Integration
The FastAPI backend serves both API endpoints (`/api/query`, `/api/courses`) and static frontend files. The frontend uses vanilla JavaScript with session management for conversation continuity.

### Tool-Based Search Pattern
The system uses Anthropic's function calling where Claude autonomously decides when to search. The `CourseSearchTool` provides semantic search with optional course name and lesson number filtering. Search sources are tracked and returned to the frontend for transparency.

### Session and Context Management
Each query can include a `session_id` for conversation continuity. The `SessionManager` maintains a rolling history of query-response pairs that get included in the system prompt for context-aware responses.
- always use uv to run the server. do not use pip directly