# Use the official Python image as a base image
FROM python:3.9.6

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure ipykernel is installed
RUN pip install ipykernel && \
    python -m ipykernel install --user --name=python3

# Verify matplotlib installation
RUN python -m pip show matplotlib

# Copy the rest of the application code into the container
COPY . .

# Ensure you have the necessary tools installed for unzipping
RUN apt-get update && apt-get install -y unzip

# Create Jupyter configuration file and disable authentication
RUN jupyter notebook --generate-config && \
    echo "c.ServerApp.token = ''" >> /root/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.password = ''" >> /root/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.disable_check_xsrf = True" >> /root/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.allow_remote_access = True" >> /root/.jupyter/jupyter_server_config.py

# Expose any ports if necessary (e.g., for Jupyter Notebook)
EXPOSE 8888

# Command to start a Jupyter notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
