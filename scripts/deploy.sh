#!/bin/bash

echo "Deploying Federated Learning on Kubernetes..."

# Create namespace
kubectl apply -f k8s/namespace.yaml

# Deploy storage
kubectl apply -f k8s/storage/

# Deploy configmaps
kubectl apply -f k8s/configmaps/

# Deploy services
kubectl apply -f k8s/services/

# Deploy server
kubectl apply -f k8s/deployments/fl-server.yaml

# Wait for server to be ready
echo "Waiting for server to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/fl-server -n federated-learning

# Deploy clients
kubectl apply -f k8s/deployments/fl-client.yaml

echo "Deployment complete!"
echo "Check status with: kubectl get pods -n federated-learning"