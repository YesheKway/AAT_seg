# Use an official Python runtime as a parent image
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev
# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install -y vim

RUN pip install -r requirements.txt

