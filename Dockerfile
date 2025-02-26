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
    build-essential \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create conda environment
RUN conda env create -f environment.yml || (cat /root/.conda/environments.txt && exit 1)

# Set up shell to use conda environment by default
SHELL ["conda", "run", "-n", "sailing-assistant", "/bin/bash", "-c"]

# Explicitly uninstall any existing PyTorch installation and install CPU-only version
RUN pip uninstall -y torch torchvision torchaudio && \
    pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

# Install custom modules
COPY custom_modules/ ./custom_modules/
# The base byaldi package is already installed via environment.yml
# Now install our custom extension as a different package name
RUN cd custom_modules/byaldi && pip install -e .

# Copy application code
COPY src/ ./src/
COPY .chainlit/ ./.chainlit/
COPY chainlit.md ./

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
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV CHAINLIT_HOST=0.0.0.0
ENV CHAINLIT_PORT=8000
ENV PYTHONUNBUFFERED=1
# Explicitly set CUDA_VISIBLE_DEVICES to -1 to ensure PyTorch uses CPU
ENV CUDA_VISIBLE_DEVICES=-1

# Start the application
CMD ["bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate sailing-assistant && chainlit run src/app.py --port 8000 --host 0.0.0.0"] 