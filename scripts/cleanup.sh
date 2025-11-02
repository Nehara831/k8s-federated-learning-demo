#!/bin/bash

echo "Cleaning up Federated Learning deployment..."

kubectl delete namespace federated-learning

echo "Cleanup complete!"