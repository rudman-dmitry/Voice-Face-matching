# Use the official Python image as a base image
FROM python:3.9.6

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Ensure you have the necessary tools installed for unzipping
RUN apt-get update && apt-get install -y unzip

# Expose any ports if necessary (e.g., for Jupyter Notebook)
EXPOSE 8888

# Command to start a Jupyter notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
