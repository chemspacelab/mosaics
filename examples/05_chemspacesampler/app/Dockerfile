# To install run: docker build -f Dockerfile -t chemspace ../../.. in examples/05_chemspacesampler/app

FROM continuumio/miniconda3

#added
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*


# Set the working directory to /app
WORKDIR /app
ADD . /app

# Install any needed packages specified in requirements.txt
COPY examples/05_chemspacesampler/app/chemspace.yml /app
COPY examples/05_chemspacesampler/app/radar.png /app

RUN conda env create -f chemspace.yml
# Make
# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "chemspace", "/bin/bash", "-c"]
# Make RUN commands use the new environment

# Make sure the environment is activated:
RUN echo "Make sure chemspace is activated:" && echo $CONDA_DEFAULT_ENV

# Activate the environment and install 'dscribe'
RUN echo "source activate chemspace" > ~/.bashrc
ENV PATH /opt/conda/envs/chemspace/bin:$PATH
RUN /bin/bash -c "source activate chemspace && conda install -c conda-forge dscribe"


# Define environment variable
ENV NAME chemspace


COPY mosaics /app
ENV PYTHONPATH "${PYTHONPATH}:/app/mosaics"
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV STREAMLIT_TELEMETRY_OPT_OUT=true
# Copy the content of the local src directory to the working directory
COPY . .



EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
# command to run on container start
ENTRYPOINT ["streamlit", "run", "examples/05_chemspacesampler/app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
#Next, if built successfully, you can run the image with:
#docker run -p 8501:8501 chemspace
