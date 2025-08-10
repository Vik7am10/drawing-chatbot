# Fresco - Architectural RAG System

A FastAPI-based RAG (Retrieval-Augmented Generation) system for querying architectural drawings and documents with vision analysis capabilities.

## Features

- 🏗️ **Document-Specific Queries**: Analyze specific architectural sheets (e.g., "show A8.4")
- 🔍 **Vision Analysis**: GPT-4o vision integration for analyzing architectural drawings
- 📊 **Multi-Modal RAG**: Combines text embeddings and visual analysis
- 🎯 **Targeted Element Analysis**: Extract and analyze specific elements from drawings
- 📱 **RESTful API**: Clean FastAPI interface with CORS support

## System Architecture

- **Backend**: FastAPI with intelligent query routing
- **RAG System**: Dual-mode text and image vector databases
- **Vision**: GPT-4o for architectural drawing analysis
- **Embeddings**: BGE text embeddings + OpenCLIP image embeddings
- **Database**: ChromaDB for vector storage

## Quick Start

### 1. Prerequisites

- Python 3.9+
- OpenAI API key with GPT-4o access

### 2. Installation

```bash
git clone <repository-url>
cd fresco
pip install -r requirements.txt
```

### 3. API Key Setup

Create an `api_key` file in the root directory:

```bash
# Create the file
touch api_key

# Add your OpenAI API key to the file
echo "your-openai-api-key-here" > api_key
```

**Important**: Never commit your `api_key` file to version control. It's already included in `.gitignore`.

### 4. Start the Server

```bash
python simple_backend.py
```

The API will be available at `http://localhost:8000`

### 5. Test the API

```bash
# Document analysis
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "show A8.4"}'

# Element-specific query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "analyze Banquet Hall in A8.4"}'

# Counting query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "how many drawings in A4.6"}'
```

## API Endpoints

### POST `/query`

Main query endpoint for architectural document analysis.

**Request Body:**
```json
{
  "query": "your query here",
  "max_results": 5
}
```

**Response:**
```json
{
  "answer": "Analysis result...",
  "confidence": 0.9,
  "context_summary": "Brief summary...",
  "source_sheets": ["A8.4"],
  "measurements": ["optional measurements"],
  "parsed_query": {},
  "image_data": "base64-encoded-image-data",
  "image_filename": "source-image.png"
}
```

## Query Types

### Document-Specific Queries
- `"show A8.4"` - Display and analyze a specific sheet
- `"analyze document A4.6"` - Comprehensive document analysis
- `"detailed analysis of A8.4"` - In-depth vision analysis

### Element-Specific Queries
- `"Banquet Hall in A8.4"` - Analyze specific room/element
- `"Kitchen details in M-2.1"` - Focus on particular spaces

### Counting Queries
- `"how many drawings in A4.6"` - Count elements in drawings
- `"number of rooms in A3.0"` - Count specific features

### Measurement Queries
- `"ceiling height in Gallery 205"` - Extract dimensions
- `"room dimensions in A8.4"` - Get measurements

## Architecture Details

### Vector Databases

- **Text Embeddings**: `embeddings_advanced_working/` - BGE-based text search
- **Image Database**: `image_vectordb/` - Source architectural images
- **Extracted Vectors**: `vectordb_extracted/` - Pre-processed embeddings

### Key Components

- `simple_backend.py` - Main FastAPI server
- `intelligent_arch_system.py` - Advanced RAG with cross-linking
- `arch_rag_system.py` - Base RAG system with dual embeddings

### Query Processing Flow

1. **Document Detection** - Identify sheet references (A8.4, M-2.1, etc.)
2. **Intent Classification** - Determine query type (vision/text/measurement)
3. **Routing Decision** - Choose between text-only or vision analysis
4. **RAG Retrieval** - Get relevant text/image embeddings
5. **Vision Analysis** - GPT-4o analysis of architectural drawings
6. **Response Generation** - Combine embeddings + vision for final answer

## Development

### Adding New Drawings

1. Add images to `annotation_images/`
2. Update vector databases with new embeddings
3. Test queries against new sheets

### Extending Query Types

1. Update intent patterns in `classify_query_intent()`
2. Add routing logic in main query handler
3. Test new query patterns

## Configuration

- **Models**: GPT-4o for vision, BGE for text embeddings
- **Vector DB**: ChromaDB with persistent storage
- **CORS**: Enabled for all origins (configure for production)

## Troubleshooting

### Common Issues

1. **"RAG system not available"**
   - Check vector database paths exist
   - Verify ChromaDB installation

2. **"Vision analysis failed"**
   - Verify OpenAI API key is valid
   - Check GPT-4o access permissions

3. **"Document not found"**
   - Ensure sheet ID format matches (e.g., "A8.4", not "A84")
   - Check image exists in `image_vectordb/`

### Logs

The system provides detailed logging for debugging:
- Document detection results
- Intent classification
- Vision analysis status
- RAG retrieval information

## License

[Add your license here]