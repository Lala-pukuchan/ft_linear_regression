# Use the official Python image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Install necessary system packages
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y poppler-utils python3-tk && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY ./requirements.txt /app/

# Install Python dependencies from requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose the port Jupyter will run on
EXPOSE 8888

# Start the Jupyter Notebook server
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]
