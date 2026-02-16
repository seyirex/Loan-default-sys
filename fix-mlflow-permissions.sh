#!/bin/bash
# Fix MLflow directory permissions for Docker container access
# Run this script if you get "Permission denied" errors when loading models

echo "Fixing MLflow directory permissions..."

# Make sure mlflow directory exists
mkdir -p mlflow

# Fix ownership and permissions for the mlflow directory
# This allows the container user (UID 1000) to read/write
sudo chown -R 1000:1000 mlflow/
sudo chmod -R 755 mlflow/

# Fix specific subdirectories
if [ -d "mlflow/1" ]; then
    sudo chmod -R 755 mlflow/1
fi

if [ -d "mlflow/artifacts" ]; then
    sudo chmod -R 755 mlflow/artifacts
fi

if [ -d "mlflow/models" ]; then
    sudo chmod -R 755 mlflow/models
fi

# Make mlflow.db writable
if [ -f "mlflow/mlflow.db" ]; then
    sudo chmod 644 mlflow/mlflow.db
fi

echo "Permissions fixed!"
echo "Now restart your services: docker-compose restart"
