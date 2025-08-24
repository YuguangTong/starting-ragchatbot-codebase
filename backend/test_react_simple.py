"""Simple ReAct integration test to verify the system works"""

import os
import tempfile
import shutil
from pathlib import Path
from config import Config
from rag_system import RAGSystem


def test_react_system():
    """Simple test to verify ReAct system works end-to-end"""
    # Create temporary directory for test database
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Configure test settings
        config = Config()
        config.CHROMA_PATH = str(Path(temp_dir) / "test_chroma")
        config.ENABLE_REACT = True
        config.MAX_REACT_ITERATIONS = 3
        config.REACT_DEBUG = True
        
        # Create RAG system
        rag_system = RAGSystem(config)
        
        # Add test documents from the existing docs folder
        docs_folder = "../docs"
        if os.path.exists(docs_folder):
            print("Adding course documents...")
            courses_added, chunks_added = rag_system.add_course_folder(docs_folder)
            print(f"Added {courses_added} courses with {chunks_added} chunks")
            
            if courses_added > 0:
                # Test a query that might trigger ReAct
                query = "Compare the main concepts taught in different programming courses"
                print(f"\nTesting query: {query}")
                
                response, sources = rag_system.query(query)
                print(f"\nResponse length: {len(response)}")
                print(f"Sources found: {len(sources)}")
                print(f"Response preview: {response[:200]}...")
                
                return True
            else:
                print("No courses found to test with")
                return False
        else:
            print("No docs folder found, skipping test")
            return False
            
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    success = test_react_system()
    if success:
        print("\n✅ ReAct system test passed!")
    else:
        print("\n❌ ReAct system test failed!")