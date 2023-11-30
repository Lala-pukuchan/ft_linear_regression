# Use the official Python base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY ./requirements.txt /app/

# Update and upgrade the system packages
# Then install any necessary system packages
# Then clean up the cache to reduce the layer size
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y poppler-utils && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel in a single RUN statement
# Install autopep8 and other Python dependencies from requirements.txt in a single RUN statement
RUN pip install --upgrade pip setuptools wheel && \
    pip install autopep8 && \
    pip install -r requirements.txt
