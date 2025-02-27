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

## Usage

1. Start the server:

```sh
pixi run start
```

or

```sh
pixi run python main.py
```

2. The API will be available at `http://localhost:7860`

3. Available endpoints:
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
