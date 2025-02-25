FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy environment.yml and requirements.txt
COPY environment.yml requirements.txt ./

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create conda environment
RUN conda env create -f environment.yml

# Set up shell to use conda environment by default
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]

# Install custom modules
COPY custom_modules/ ./custom_modules/
RUN pip install -e ./custom_modules/byaldi

# Copy application code
COPY src/ ./src/
COPY .chainlit/ ./.chainlit/
COPY chainlit.md ./
COPY .env ./.env

# Copy data directories (these will be mounted in production)
# In development, you can uncomment these lines to include the data in the image
# COPY data/ ./data/
# COPY .byaldi/ ./.byaldi/
# COPY chroma_db/ ./chroma_db/

# Create directories for mounted volumes
# Note: The application uses Path(__file__).parent.parent.absolute() to determine paths,
# so these directories must be in the expected location relative to the source files
RUN mkdir -p data/pdfs .byaldi chroma_db

# Expose the port Chainlit runs on
EXPOSE 8080

# Set environment variables
ENV PYTHONPATH=/app
ENV CHAINLIT_HOST=0.0.0.0
ENV CHAINLIT_PORT=8080

# Start the application
CMD ["conda", "run", "--no-capture-output", "-n", "base", "python", "-m", "chainlit", "run", "src/app.py"] 