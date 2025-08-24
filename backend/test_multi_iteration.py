"""Test script to demonstrate multi-iteration ReAct behavior"""

import os
from config import Config, config
from rag_system import RAGSystem


def test_multi_iteration_react():
    """Test queries designed to trigger multiple ReAct iterations"""
    
    # Enable debug to see iterations
    config.REACT_DEBUG = True
    config.ENABLE_REACT = True
    config.MAX_REACT_ITERATIONS = 5
    
    # Create RAG system
    rag_system = RAGSystem(config)
    
    # Add documents if not already present
    docs_folder = "../docs"
    if os.path.exists(docs_folder):
        courses_added, chunks_added = rag_system.add_course_folder(docs_folder)
        print(f"Loaded {courses_added} courses with {chunks_added} chunks")
    
    print("\n" + "="*80)
    print("TESTING MULTI-ITERATION REACT SCENARIOS")
    print("="*80)
    
    # Test 1: Deliberate multi-step query with explicit instructions
    print("\n🧪 Test 1: Explicit multi-step search instructions")
    query1 = """I need you to do this in steps:
1. First search for 'MCP tools' to understand tool concepts
2. Then search for 'Chroma embedding' to understand embedding concepts  
3. Then search for 'computer use workflow' to understand automation
4. Finally, explain how these three concepts work together in AI applications"""
    
    print(f"Query: {query1}")
    print("\nExecuting...")
    response1, sources1 = rag_system.query(query1)
    print(f"Response length: {len(response1)}")
    print(f"Sources found: {len(sources1)}")
    
    # Test 2: Comparative analysis requiring multiple searches
    print("\n🧪 Test 2: Cross-course concept comparison")
    query2 = """Compare and contrast how each course approaches AI integration:
- Find MCP's approach to AI integration
- Find Chroma's approach to AI integration  
- Find Computer Use's approach to AI integration
- Then provide a detailed comparison table"""
    
    print(f"Query: {query2}")
    print("\nExecuting...")
    response2, sources2 = rag_system.query(query2)
    print(f"Response length: {len(response2)}")
    print(f"Sources found: {len(sources2)}")
    
    # Test 3: Research-then-analyze pattern
    print("\n🧪 Test 3: Research-then-analyze pattern")
    query3 = """Research all mentions of 'vector search' across all courses, then for each course that mentions it, find what specific vector search techniques they teach, then analyze which course has the most comprehensive vector search coverage"""
    
    print(f"Query: {query3}")
    print("\nExecuting...")
    response3, sources3 = rag_system.query(query3)
    print(f"Response length: {len(response3)}")
    print(f"Sources found: {len(sources3)}")
    
    print("\n" + "="*80)
    print("MULTI-ITERATION REACT TESTING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    test_multi_iteration_react()