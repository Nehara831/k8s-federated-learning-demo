#!/bin/bash

# Quick redeploy script - rebuilds images and redeploys to K8s
# Usage: ./scripts/redeploy.sh

set -e  # Exit on error

echo "================================================"
echo "  FL System - Complete Rebuild & Redeploy"
echo "================================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

# Step 1: Clean up old deployment
echo "[1/5] Cleaning up old deployment..."
kubectl delete namespace federated-learning --ignore-not-found=true
echo "Waiting for namespace to be fully deleted..."
sleep 10

# Step 2: Rebuild Docker images
echo ""
echo "[2/5] Rebuilding Docker images..."
chmod +x scripts/build-images.sh
./scripts/build-images.sh

# Step 3: Verify images
echo ""
echo "[3/5] Verifying Docker images..."
if docker images | grep -q "fl-server.*latest" && docker images | grep -q "fl-client.*latest"; then
    echo "✅ Images built successfully:"
    docker images | grep fl-
else
    echo "❌ Error: Images not found!"
    exit 1
fi

# Step 4: Deploy to Kubernetes
echo ""
echo "[4/5] Deploying to Kubernetes..."
chmod +x scripts/deploy.sh
./scripts/deploy.sh

# Step 5: Wait for pods to be ready
echo ""
echo "[5/5] Waiting for pods to be ready..."
echo "Waiting for namespace..."
kubectl wait --for=condition=Active namespace/federated-learning --timeout=30s

echo "Waiting for server pod..."
kubectl wait --for=condition=Ready pod -l app=fl-server -n federated-learning --timeout=120s || true

echo "Waiting for client pods..."
kubectl wait --for=condition=Ready pod -l app=fl-client -n federated-learning --timeout=120s || true

# Show status
echo ""
echo "================================================"
echo "  Deployment Complete!"
echo "================================================"
echo ""
echo "📊 Current Status:"
kubectl get pods -n federated-learning

echo ""
echo "📝 Useful Commands:"
echo "  - Watch server logs:  kubectl logs -f -n federated-learning -l app=fl-server"
echo "  - Watch all pods:     kubectl get pods -n federated-learning -w"
echo "  - Check client logs:  kubectl logs -n federated-learning fl-client-0"
echo "  - Delete deployment:  kubectl delete namespace federated-learning"
echo ""
echo "🔍 Check for fixes:"
echo "  - IID partitioning:   kubectl logs -n federated-learning -l app=fl-server | grep 'Partition lengths'"
echo "  - No overflow errors: kubectl logs -n federated-learning -l app=fl-server | grep -i 'overflow\\|infinity'"
echo "  - Detector working:   kubectl logs -n federated-learning -l app=fl-server | grep -i 'detector\\|malicious'"
echo ""
