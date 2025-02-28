# PDF and Image AI Mapper

A Python backend for processing, categorizing, and searching through PDF documents and images using OCR.

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

   - `POST /upload/`: Upload and process a PDF or image file
   - `POST /search/`: Search through processed documents
   - `GET /categories/`: Get all available document categories

## API Reference

### Upload a Document

```
POST /upload/
```

**Request Body**: Form data with a file field

**Response**:
```json
{
  "status": "success",
  "message": "File processed successfully",
  "document_id": "uuid",
  "categories": ["category1", "category2"]
}
```

### Search Documents

```
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

```
GET /categories/
```

**Response**:
```json
{
  "categories": ["category1", "category2", "category3"]
}
```

## How It Works

1. **Document Processing**:
   - PDF files are parsed for text using PyPDF2
   - If text extraction fails or is limited, PDF pages are converted to images for OCR
   - Images are processed using Tesseract OCR
   - Extracted text is cleaned and preprocessed

2. **Categorization**:
   - Documents are categorized using unsupervised K-means clustering
   - TF-IDF vectorization is used to represent document content
   - Category names are generated from important terms in each cluster

3. **Search**:
   - Search uses a simple but effective relevance scoring system
   - Results can be filtered by categories
   - Relevant snippets are generated to show matching context

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
