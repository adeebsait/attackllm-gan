# Use a lightweight base image
FROM python:3.9-alpine

# Set the working directory
WORKDIR /app

# Install necessary packages
# - Flask for the web server
# - curl for downloading the agent
RUN pip install Flask

# Copy the Flask application into the container
COPY app.py .

# Expose the port the app runs on
EXPOSE 80

# Define environment variable for the CALDERA server address
# This will be passed in from docker-compose.yml
ENV CALDERA_SERVER=http://127.0.0.1:8888

# Command to run the application
# 1. Download the Sandcat (54ndc47) agent for Linux
# 2. Make it executable
# 3. Run the agent in the background, connecting to the CALDERA server
# 4. Start the vulnerable Flask web server
CMD wget -q $CALDERA_SERVER/file/download -O 54ndc47.go && \
    chmod +x 54ndc47.go && \
    ./54ndc47.go -server $CALDERA_SERVER -group red & \
    python app.py
