# Use the official Python 3.9 image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt /app/

# Install the dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app/

# Expose the port your app will run on (assuming Flask/Django app listens on port 5000)
EXPOSE 5000

# Command to run the app with Gunicorn (for production use)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT"]

# For development (uncomment the following line if you're in development mode):
# CMD ["python", "app.py"]
