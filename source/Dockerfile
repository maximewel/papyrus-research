FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Set up a working directory
WORKDIR /workspace

# Create and activate a virtual environment
RUN python3 -m venv /workspace/venv
ENV PATH="/workspace/venv/bin:$PATH"


# Install the necessary OpenGL and glib dependencies for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Install Python dependencies
COPY requirements_docker.txt .
RUN pip install --upgrade pip && pip install -r requirements_docker.txt

# Copy the entire workspace into the container
COPY . /workspace/source

# Default command to run your module (can be overridden in docker-compose.yml)
CMD ["python", "-m", "source.run_experiment"]