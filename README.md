# PDF and Image AI Mapper

A tool for processing, categorizing, and searching through PDF documents and images using OCR.

## Features

- Upload and process PDF documents and images
- Extract text using OCR (Optical Character Recognition)
- Automatically categorize documents based on content
- Search through processed documents
- Filter search results by categories

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
- `POST /recategorize/`: Manually trigger recategorization of all documents (optional)
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
  "message": "File uploaded successfully and processing started (categorization will happen automatically)",
  "document_id": "<uuid>"
  "categories":["Processing"]
}
```

### Search Documents

```text
POST /search/
```

**Request Body**:

```json
{
  "query": "search query",
  "categories": ["optional_category1"]
}
```

**Response**:

```json
{
  "results": [
    {
      "document_id": "uuid",
      "filename": "document.pdf",
      "categories": ["category1"],
      "score": 10,
      "snippet": "...matching text snippet..."
    }
  ]
}
```

### Get Categories

```text
GET /categories/
```

**Response**:

```json
{
  "categories": ["category1", "category2", "category3"]
}
```

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
      "categories": ["category1", "category2"]
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

## How It Works

1. **Document Processing**:
   - PDF files are parsed for text using PyPDF2
   - If text extraction fails or is limited, PDF pages are converted to images for OCR
   - Images are processed using Tesseract OCR
   - Extracted text is cleaned and preprocessed

2. **Categorization**:
   - Documents are automatically categorized after upload using unsupervised K-means clustering
   - TF-IDF vectorization is used to represent document content
   - Category names are generated from important terms in each cluster
   - The categorization process runs in the background after each document is processed

3. **Search**:
   - Search uses a simple but effective relevance scoring system
   - Results can be filtered by categories
   - Relevant snippets are generated to show matching context

## Document Processing Workflow

1. User uploads a file via the `/upload/` endpoint
2. File is saved and processing begins in the background
3. Text is extracted from the document using PDF parsing or OCR
4. Document is added to the search index
5. Automatic recategorization is triggered for all documents
6. Categories are updated and saved
7. The updated categories are available via the `/categories/` endpoint

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
