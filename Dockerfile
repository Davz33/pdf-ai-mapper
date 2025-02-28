FROM mcr.microsoft.com/devcontainers/python:3.12

# Install system dependencies (Tesseract OCR and Poppler for PDF to image conversion)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install pixi and make it available in PATH
RUN curl -fsSL https://pixi.sh/install.sh | bash && \
    echo 'export PATH="/root/.pixi/bin:$PATH"' >> ~/.bashrc

# Add pixi to PATH for the current session and install dependencies
ENV PATH="/root/.pixi/bin:$PATH"
RUN /root/.pixi/bin/pixi install

# Expose port
EXPOSE 7860

# Set default command
CMD ["/root/.pixi/bin/pixi", "run", "start"] 