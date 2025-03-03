# PDF and Image AI Mapper

A Python backend for processing, categorizing, and searching through PDF documents and images using OCR.

## Features

- Upload and process PDF documents and images
- Extract text using OCR (Optical Character Recognition)
- Automatically categorize documents based on content
- Search through processed documents
- Filter search results by categories
- Detect and handle duplicate documents

## Requirements

- Python 3.8+
- Tesseract OCR engine
- Poppler (for PDF to image conversion)
- Pixi package manager

## Installation

### Option 1: Local Installation

1. Install Tesseract OCR engine:
   - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
   - **MacOS**: `brew install tesseract`
   - **Windows**: Download and install from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

2. Install Poppler:
   - **Ubuntu/Debian**: `sudo apt-get install poppler-utils`
   - **MacOS**: `brew install poppler`
   - **Windows**: Download and install from [poppler releases](https://github.com/oschwartz10612/poppler-windows/releases/)

3. Install [Pixi](https://pixi.sh) if you haven't already:

```sh
curl -fsSL https://pixi.sh/install.sh | bash
```

4. Create a development environment and install dependencies:

```sh
pixi install
```

### Option 2: Using Docker

1. Make sure you have Docker and Docker Compose installed:
   - [Install Docker](https://docs.docker.com/get-docker/)
   - [Install Docker Compose](https://docs.docker.com/compose/install/)

2. Build and run the Docker container:

```sh
docker-compose up --build
```

This will:

- Build the Docker image with all required dependencies
- Start the application on port 7860
- Mount volumes for uploads and processed data

## Usage

### Local Usage

1. Start the server:

```sh
pixi run start
```

or

```sh
pixi run python main.py
```

### Docker Usage

1. Start the application with Docker Compose:

```sh
docker-compose up
```

2. To run in the background:

```sh
docker-compose up -d
```

3. To stop the application:

```sh
docker-compose down
```

The API will be available at `http://localhost:7860` for both local and Docker usage.

## Available Endpoints

- `POST /upload/`: Upload and process a PDF or image file (automatically categorizes documents)
- `POST /search/`: Search through processed documents
- `GET /categories/`: Get all available document categories
- `POST /recategorize/`: Manually trigger recategorization of all documents (optional, as categorization happens automatically)
- `POST /recategorize-with-clusters/?clusters=<number>`: Manually trigger recategorization with a custom number of clusters
- `POST /cleanup-duplicates/`: Remove duplicate documents from the index
- `GET /status/`: Check the processing status of all documents

## API Reference

### Upload a Document

```text
POST /upload/
```

For example, using curl:

```bash
curl -X POST http://localhost:7860/upload/ -F "file=@<path-to-pdf.pdf>"
```

**Request Body**: Form data with a file field

**Response**:

```json
{
  "status": "success",
  "message": "File uploaded successfully and processing started (categorization will happen automatically, duplicates will be detected)",
  "document_id": "<uuid>",
  "categories": ["Processing"]
}
```

### Search Documents

```text
POST /search/
```

This endpoint allows you to search through all processed documents using a text query. You can optionally filter results by specific categories.

For example, using curl:

```bash
curl -X POST http://localhost:7860/search/ -H "Content-Type: application/json" -d '{"query": "ai"}'
```

To filter by category:

```bash
curl -X POST http://localhost:7860/search/ -H "Content-Type: application/json" -d '{"query": "ai", "categories": ["Analysis: ai, data, industries"]}'
```

**Request Body**:

```json
{
  "query": "search query",
  "categories": ["optional_category1"]  // Optional array of categories to filter by
}
```

**Response**:

```json
{
  "results": [
    {
      "document_id": "uuid",
      "filename": "document.pdf",
      "categories": ["Document: ai, data, industries"],
      "score": 684,
      "snippet": "...matching text snippet with highlighted search terms..."
    }
  ]
}
```

The results are sorted by relevance score, with higher scores indicating better matches. The snippet shows the context where the search terms appear in the document.

**Note:** The system automatically detects and handles duplicate documents. If the same document is uploaded multiple times, the system will recognize it and use the existing document ID, preventing duplicate entries in search results.

### Get Categories

```text
GET /categories/
```

This endpoint returns all available document categories in the system.

For example, using curl:

```bash
curl http://localhost:7860/categories/
```

**Response**:

```json
{
  "categories": [
    "Document: ai, data, industries",
    "Report: review, time, problem",
    "Analysis: ai, data, industries",
    "Research: ai, data, industries"
  ]
}
```

Each category has a descriptive prefix (Document, Report, Analysis, etc.) followed by the most important terms that characterize that category.

### Check Processing Status

```text
GET /status/
```

This endpoint allows you to check the processing status of all documents in the system.

For example, using curl:

```bash
curl http://localhost:7860/status/
```

**Response**:

```json
{
  "documents": [
    {
      "document_id": "<uuid>",
      "filename": "document.pdf",
      "status": "processed",
      "categories": ["<Cagegory naming e.g., Document: ai, data, industries>: <category 1>,<category 2>,<category 3>,..."]
    },
    {
      "document_id": "<uuid>",
      "filename": "another-document.pdf",
      "status": "processing",
      "categories": ["Processing"]
    }
  ]
}
```

Possible status values:

- `processing`: Document is still being processed
- `processed`: Document has been fully processed and categorized
- `error`: An error occurred during processing

### Manual Recategorization

To manually trigger recategorization of all documents:

```text
POST /recategorize/
```

This endpoint processes all existing documents, applies the categorization logic, updates the document index and returns the new categories. Note that this is typically not needed as categorization happens automatically after each document upload.

### Custom Recategorization with Specific Cluster Count

To manually trigger recategorization with a custom number of clusters:

```text
POST /recategorize-with-clusters/?clusters=10
```

This endpoint allows you to specify how many distinct categories you want the system to create. The `clusters` parameter can be set between 2 and 20, with a default of 8 if not specified. If the number of clusters exceeds the number of documents, the system will automatically adjust the cluster count to match the document count.

**Request Parameters**:
- `clusters`: Integer between 2 and 20 (default: 8)

**Response**:

```json
{
  "status": "success",
  "message": "All documents recategorized",
  "categories": ["Document: ai, data, industries", "Report: review, time, problem", ...]
}
```

This is useful when you want more granular categories (higher number) or broader categories (lower number).

### Cleanup Duplicates

To remove duplicate documents from the index:

```text
POST /cleanup-duplicates/
```

This endpoint identifies and removes duplicate documents based on content hash, keeping only one copy of each unique document.

For example, using curl:

```bash
curl -X POST http://localhost:7860/cleanup-duplicates/
```

**Response**:

```json
{
  "status": "success",
  "message": "Removed 3 duplicate documents",
  "document_count": 5
}
```

This is useful for cleaning up the document index if the same documents were uploaded multiple times before duplicate detection was implemented.

## How It Works

1. **Document Processing**:
   - PDF files are parsed for text using PyPDF2
   - If text extraction fails or is limited, PDF pages are converted to images for OCR
   - Images are processed using Tesseract OCR
   - Extracted text is cleaned and preprocessed
   - Content hashing is used to detect duplicate documents

2. **Categorization**:
   - Documents are automatically categorized after upload using unsupervised K-means clustering
   - TF-IDF vectorization is used to represent document content
   - Category names are generated from important terms in each cluster
   - Categories use descriptive prefixes (Document, Report, Analysis, etc.) for better readability
   - The system ensures categories are unique and descriptive
   - By default, the system creates 8 distinct categories (can be customized)
   - The categorization process runs in the background after each document is processed
   - The system automatically adjusts the number of clusters if there are fewer documents than requested clusters

3. **Search**:
   - Search uses a simple but effective relevance scoring system
   - Results can be filtered by categories
   - Relevant snippets are generated to show matching context
   - Duplicate documents are automatically filtered from search results

4. **Duplicate Detection**:
   - The system calculates a content hash for each document
   - When a new document is uploaded, it's compared against existing documents
   - If a duplicate is found (by filename or content), the existing document is used
   - This prevents duplicate entries in the document index and search results

## Document Processing Workflow

1. User uploads a file via the `/upload/` endpoint
2. File is saved and processing begins in the background
3. Text is extracted from the document using PDF parsing or OCR
4. Document is added to the search index
5. Automatic recategorization is triggered for all documents
6. Categories are updated and saved with descriptive names
7. The updated categories are available via the `/categories/` endpoint
8. Document status can be checked via the `/status/` endpoint

## Logging from Running Docker Container

This application will log errors upon API calls, which you can visualize via:

```bash
docker logs -ft --tail 0 <running-container-name>
```

## Getting the GitHub Codespace Docker Image

GitHub Codespaces doesn't provide a direct way to download the exact Docker image it uses. However, the Dockerfile in this project replicates the functionality of the GitHub Codespace by:

1. Using the same base image (`mcr.microsoft.com/devcontainers/python:3.12`)
2. Installing the same system dependencies
3. Setting up pixi in the same way as the devcontainer.json

This approach ensures that the Docker environment closely matches the GitHub Codespace environment.

## Known Limitations and Future Improvements

### Duplicate Document Handling

The system now implements duplicate detection and handling:

- Documents are identified by both filename and content hash
- If a duplicate document is uploaded, the system uses the existing document ID
- Search results automatically filter out duplicate content
- This prevents redundant processing and improves search result quality

### Other Planned Enhancements

- Improved error handling for malformed documents
- Support for more document formats (e.g., DOCX, XLSX)
- Enhanced search with semantic capabilities
- User authentication and document ownership
- Web interface for easier interaction
