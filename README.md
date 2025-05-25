# PDF and Image AI Mapper

A full-stack tool for processing, categorizing, and searching through PDF documents and images using OCR.

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
   - Support for structured categories with metadata for enterprise applications
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

## Monorepo Setup

This project uses [pnpm workspaces](https://pnpm.io/workspaces) and [Nx](https://nx.dev/) for managing the frontend, and [Pixi](https://pixi.sh) for the Python backend.

```text
root/
├── apps/
│   ├── backend-python/      # Python backend (FastAPI/Flask, managed by Pixi)
│   └── frontend/            # React app (Nx + pnpm)
├── libs/
│   └── protos/              # Shared .proto files for gRPC (if any)
├── package.json
├── pnpm-workspace.yaml
├── nx.json
├── tsconfig.base.json
├── .gitignore
├── README.md
└── ...
```

## Requirements

- Python (for backend)
- Node.js (for frontend)
- pnpm (for monorepo and frontend)
- Pixi package manager (for backend environment)
- Docker (optional, for containerized running and development)
- Tesseract OCR engine
- Poppler (for PDF to image conversion)

## Features

- Upload and process PDF documents and images
- Extract text using OCR (Optical Character Recognition)
- Automatically categorize documents based on content
- Search through processed documents
- Filter search results by categories
- Detect and handle duplicate documents
- Support for structured categories with metadata for enterprise applications

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

4. Install pnpm

    ```sh
    npm install -g pnpm
    ```

5. Install monorepo dependencies

    ```sh
    pnpm install
    ```

6. Install Backend Python dependencies:

    ```sh
    cd apps/backend-python && pixi install
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

1. Start the Python Backend:

    ```sh
    cd apps/backend-python && pixi run start
    ```

2. Start the React Frontend:

    ```sh
    pnpm nx serve frontend
    ```

or (if using Vite directly) `cd apps/frontend && pnpm run dev`

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
- `POST /generate-structured-categories/`: Generate structured categories from existing categories
- `GET /status/`: Check the processing status of all documents

## API Reference

### Upload a Document

```text
POST /upload/
```

For example, using curl:

```bash
curl -X POST http://localhost:7860/upload/ -F "file=@<path_to_file.pdf>"
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

This endpoint allows you to search through all processed documents using a text query. You can optionally filter results by specific categories, category types, or keywords.

For example, using curl:

```bash
curl -X POST http://localhost:7860/search/ -H "Content-Type: application/json" -d '{"query": "<search_query_text>"}'
```

To filter by category:

```bash
curl -X POST http://localhost:7860/search/ -H "Content-Type: application/json" -d '{"query": "<search_query_text>", "categories": ["<category_name_1>", "<category_name_2>"]}'
```

To filter by category type:

```bash
curl -X POST http://localhost:7860/search/ -H "Content-Type: application/json" -d '{"query": "<search_query_text>", "category_types": ["<category_type_1>", "<category_type_2>"]}'
```

To filter by keywords:

```bash
curl -X POST http://localhost:7860/search/ -H "Content-Type: application/json" -d '{"query": "<search_query_text>", "keywords": ["<keyword_1>", "<keyword_2>"]}'
```

**Request Body**:

```json
{
  "query": "<search_query_text>",
  "categories": ["<optional_category_name_1>", "<optional_category_name_2>"],  // Optional array of categories to filter by
  "category_types": ["<optional_category_type_1>", "<optional_category_type_2>"],  // Optional array of category types to filter by
  "keywords": ["<optional_keyword_1>", "<optional_keyword_2>"]  // Optional array of keywords to filter by
}
```

**Response**:

```json
{
  "results": [
    {
      "document_id": "<uuid>",
      "filename": "<original_filename.pdf>",
      "categories": ["<category_type>: <keyword_1>, <keyword_2>, <keyword_3>"],
      "structured_categories": [
        {
          "id": "<cat-XXX>",
          "type": "<category_type>",
          "keywords": ["<keyword_1>", "<keyword_2>", "<keyword_3>"],
          "display_name": "<category_type>: <keyword_1>, <keyword_2>, <keyword_3>",
          "created_at": "<ISO8601_timestamp>"
        }
      ],
      "score": <relevance_score_0_to_1000>,
      "snippet": "...matching text snippet with highlighted search terms..."
    }
  ],
  "available_filters": {
    "category_types": ["<category_type_1>", "<category_type_2>", "<category_type_3>"],
    "keywords": ["<keyword_1>", "<keyword_2>", "<keyword_3>"]
  }
}
```

The results are sorted by relevance score, with higher scores indicating better matches. The snippet shows the context where the search terms appear in the document. The `available_filters` section provides all available category types and keywords that can be used for filtering in subsequent searches.

**Note:** The system automatically detects and handles duplicate documents. If the same document is uploaded multiple times, the system will recognize it and use the existing document ID, preventing duplicate entries in search results.

### Get Categories

```text
GET /categories/
```

This endpoint returns all available document categories in the system as structured categories.

For example, using curl:

```bash
curl http://localhost:7860/categories/
```

**Response**:

```json
{
  "structured_categories": [
    {
      "id": "<cat-XXX_format>",
      "type": "<category_type_1>",
      "keywords": ["<keyword_1>", "<keyword_2>", "<keyword_3>"],
      "display_name": "<category_type_1>: <keyword_1>, <keyword_2>, <keyword_3>",
      "created_at": "<ISO8601_timestamp>"
    },
    {
      "id": "<cat-XXX_format>",
      "type": "<category_type_2>",
      "keywords": ["<keyword_4>", "<keyword_5>", "<keyword_6>"],
      "display_name": "<category_type_2>: <keyword_4>, <keyword_5>, <keyword_6>",
      "created_at": "<ISO8601_timestamp>"
    }
  ]
}
```

Each structured category includes an ID, type, keywords array, display name, and creation timestamp.

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
      "filename": "<original_filename.pdf>",
      "status": "processed",
      "categories": ["<category_type>: <keyword_1>, <keyword_2>, <keyword_3>"]
    },
    {
      "document_id": "<uuid>",
      "filename": "<another_filename.pdf>",
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

This endpoint processes all existing documents, applies the categorization logic, updates the document index and returns the new structured categories. Note that this is typically not needed as categorization happens automatically after each document upload.

**Response**:

```json
{
  "status": "success",
  "message": "Recategorized X of Y documents",
  "structured_categories": [
    {
      "id": "<cat-XXX_format>",
      "type": "<category_type_1>",
      "keywords": ["<keyword_1>", "<keyword_2>", "<keyword_3>"],
      "display_name": "<category_type_1>: <keyword_1>, <keyword_2>, <keyword_3>",
      "created_at": "<ISO8601_timestamp>"
    },
    {
      "id": "<cat-XXX_format>",
      "type": "<category_type_2>",
      "keywords": ["<keyword_4>", "<keyword_5>", "<keyword_6>"],
      "display_name": "<category_type_2>: <keyword_4>, <keyword_5>, <keyword_6>",
      "created_at": "<ISO8601_timestamp>"
    }
  ]
}
```

### Custom Recategorization with Specific Cluster Count

To manually trigger recategorization with a custom number of clusters:

```text
POST /recategorize-with-clusters/?clusters=<number_2_to_20>
```

This endpoint allows you to specify how many distinct categories you want the system to create. The `clusters` parameter can be set between 2 and 20, with a default of 8 if not specified. If the number of clusters exceeds the number of documents, the system will automatically adjust the cluster count to match the document count.

**Request Parameters**:

- `clusters`: Integer between 2 and 20 (default: 8)

**Response**:

```json
{
  "status": "success",
  "message": "All documents recategorized with X clusters",
  "structured_categories": [
    {
      "id": "<cat-XXX_format>",
      "type": "<category_type_1>",
      "keywords": ["<keyword_1>", "<keyword_2>", "<keyword_3>"],
      "display_name": "<category_type_1>: <keyword_1>, <keyword_2>, <keyword_3>",
      "created_at": "<ISO8601_timestamp>"
    },
    {
      "id": "<cat-XXX_format>",
      "type": "<category_type_2>",
      "keywords": ["<keyword_4>", "<keyword_5>", "<keyword_6>"],
      "display_name": "<category_type_2>: <keyword_4>, <keyword_5>, <keyword_6>",
      "created_at": "<ISO8601_timestamp>"
    }
  ]
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
  "message": "Removed <number_of_duplicates> duplicate documents",
  "document_count": <remaining_document_count>
}
```

This is useful for cleaning up the document index if the same documents were uploaded multiple times before duplicate detection was implemented.

### Generate Structured Categories

To generate structured categories from existing categories:

```text
POST /generate-structured-categories/
```

This endpoint converts the existing string-based categories into a structured format with type, keywords, and metadata.

For example, using curl:

```bash
curl -X POST http://localhost:7860/generate-structured-categories/
```

**Response**:

```json
{
  "status": "success",
  "message": "Generated <number_of_categories> structured categories",
  "structured_categories": [
    {
      "id": "<cat-XXX>",
      "type": "<category_type>",
      "keywords": ["<keyword_1>", "<keyword_2>", "<keyword_3>"],
      "display_name": "<category_type>: <keyword_1>, <keyword_2>, <keyword_3>",
      "created_at": "<ISO8601_timestamp>"
    },
    ...
  ]
}
```

This structured format provides better organization and more metadata for enterprise applications.

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
