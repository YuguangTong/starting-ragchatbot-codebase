# RAG System Test Suite

This directory contains comprehensive tests for the RAG system's CourseSearchTool and content-query handling capabilities.

## 🧪 Test Overview

The test suite consists of **32 tests** that validate:
- CourseSearchTool.execute method functionality
- RAG system integration and pipeline processing
- Content query evaluation and search quality metrics

### Test Results Summary
- ✅ **100% Success Rate** - All 32 tests pass
- ⚡ **Fast Execution** - Runs in ~2.3 seconds
- 📊 **Comprehensive Coverage** - Unit, integration, and quality tests

## 📁 File Structure

```
tests/
├── README.md                     # This documentation
├── __init__.py                   # Package initialization
├── fixtures.py                   # Test data and mock scenarios
├── test_helpers.py              # Mock classes and utilities
├── test_course_search_tool.py   # Unit tests (19 tests)
├── test_rag_integration.py      # Integration tests (13 tests)
├── run_tests.py                 # Test runner script
└── test_analysis_report.md      # Detailed analysis report
```

## 🚀 Running Tests

### Run All Tests
```bash
# From backend/ directory
uv run python tests/run_tests.py

# Or run individual test files
uv run python tests/test_course_search_tool.py
uv run python tests/test_rag_integration.py
```

### Run Specific Tests
```bash
# Run specific test module
python tests/run_tests.py --module test_course_search_tool

# Run specific test class
python tests/run_tests.py --module test_course_search_tool --class TestCourseSearchToolExecute

# Run specific test method
python tests/run_tests.py --module test_course_search_tool --class TestCourseSearchToolExecute --method test_basic_query_no_filters
```

## 🧪 Test Categories

### 1. CourseSearchTool Unit Tests (19 tests)
**File: `test_course_search_tool.py`**

#### Core Functionality
- ✅ Basic query processing without filters
- ✅ Course name and lesson number filtering (individual & combined)
- ✅ Fuzzy course name matching (e.g., "MCP" → "Introduction to MCP")
- ✅ Case-insensitive search handling

#### Error Handling
- ✅ Empty search results with appropriate messages
- ✅ Database/vector store error handling
- ✅ Invalid course name validation
- ✅ Proper error message formatting for different filter combinations

#### Output Quality
- ✅ Result formatting with course/lesson headers
- ✅ Source tracking for UI display
- ✅ Lesson link integration
- ✅ Multiple result ordering and separation

#### Interface Compliance
- ✅ Tool interface implementation validation
- ✅ Anthropic tool definition structure
- ✅ Proper initialization with dependencies

### 2. RAG System Integration Tests (8 tests)
**File: `test_rag_integration.py`**

#### Pipeline Integration
- ✅ End-to-end query processing pipeline
- ✅ Component communication (RAG system ↔ AI generator ↔ tools)
- ✅ Session management and conversation history
- ✅ Tool availability and exposure to AI

#### Error Resilience
- ✅ Graceful error handling when components fail
- ✅ Edge case handling (empty queries, None inputs)

### 3. Content Query Evaluation Tests (5 tests)
**File: `test_rag_integration.py`**

#### Query Type Handling
- ✅ **Factual Queries**: "What is machine learning?"
- ✅ **Procedural Queries**: "How to set up MCP?"
- ✅ **Comparative Queries**: "What's the difference between X and Y?"
- ✅ **Contextual Queries**: Course/lesson-specific questions

#### Search Quality Metrics
- ✅ Result relevance to query keywords
- ✅ Response completeness (substantial content)
- ✅ Proper source attribution
- ✅ Query specificity impact on results

## 🔧 Test Infrastructure

### Mock Components

#### MockVectorStore
- **Purpose**: Test CourseSearchTool without database dependencies
- **Features**: 
  - Simulates vector search with realistic scenarios
  - Supports fuzzy course name matching
  - Handles error scenarios (empty results, database errors)
  - Tracks search calls for verification

#### Test Fixtures
- **Sample Courses**: 3 courses (ML, MCP, Python) with lessons
- **Sample Chunks**: 7 content chunks across different courses
- **Test Scenarios**: Predefined search scenarios for consistent testing

### Test Helpers
- **MockSearchResultsBuilder**: Creates realistic SearchResults objects
- **Assertion Helpers**: Validate result formatting and structure
- **Content Extraction**: Parse and analyze search result content

## 📊 Key Test Scenarios

### Search Functionality Tests
```python
# Basic search without filters
test_basic_query_no_filters()

# Course-specific search
test_query_with_course_filter()

# Lesson-specific search  
test_query_with_lesson_filter()

# Combined filters
test_query_with_both_filters()
```

### Error Handling Tests
```python
# Empty results
test_empty_results_*()

# Database errors
test_error_handling()

# Invalid course names
test_course_not_found_error()
```

### Quality Evaluation Tests
```python
# Content relevance
test_result_relevance()

# Response completeness
test_result_completeness()

# Source attribution
test_source_attribution()
```

## 🎯 Quality Indicators

The tests validate these quality metrics:

### Search Result Quality
- ✅ **Relevance**: Results contain query-related keywords
- ✅ **Completeness**: Responses have substantial content (>10 words)
- ✅ **Attribution**: Proper course/lesson source headers
- ✅ **Formatting**: Clean structure with clear separation

### System Robustness
- ✅ **Error Resilience**: Graceful handling of failures
- ✅ **Edge Cases**: Proper validation of unusual inputs  
- ✅ **Integration**: Seamless component communication
- ✅ **Consistency**: Reliable behavior across scenarios

## 📈 Test Insights

### System Strengths Validated
1. **Robust Search**: Handles diverse query types effectively
2. **Flexible Filtering**: Course name and lesson number filters work seamlessly
3. **Error Handling**: Comprehensive error scenarios covered
4. **Result Quality**: High-quality, well-formatted responses
5. **Integration**: Strong component interaction and communication

### Coverage Areas
- **Unit Level**: Individual method functionality
- **Integration Level**: Component interaction and data flow
- **Quality Level**: Content relevance and user experience
- **Error Level**: Failure modes and recovery

## 🔍 Usage Examples

### Testing New Features
```python
# Add test to appropriate file
def test_new_feature(self):
    """Test description"""
    # Setup
    result = self.search_tool.execute("test query")
    
    # Assertions
    self.assertIn("expected content", result)
```

### Adding Mock Scenarios
```python
# In test_helpers.py
class MockVectorStore:
    def __init__(self, scenario: str = "your_scenario"):
        # Add new scenario handling
```

### Custom Test Data
```python
# In fixtures.py
SAMPLE_CHUNKS.append(CourseChunk(
    content="Your test content",
    course_title="Your Course",
    lesson_number=1,
    chunk_index=7
))
```

## 📋 Maintenance Notes

### Running Tests Regularly
- Tests should be run before any code changes
- All tests must pass before committing changes
- New features should include corresponding tests

### Mock Data Updates
- Keep sample data realistic and representative
- Update mock scenarios when adding new functionality
- Maintain consistency between fixtures and actual data structures

### Performance Monitoring
- Current execution time: ~2.3 seconds for all 32 tests
- Monitor for performance regressions as test suite grows
- Consider parallel execution for larger test suites

---

For detailed analysis and insights, see [test_analysis_report.md](test_analysis_report.md).