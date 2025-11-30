#!/usr/bin/env bash
# Optimized Minikube deployment with preprocessing and caching
set -euo pipefail

NAMESPACE=${NAMESPACE:-federated-learning}
CLIENT_IMAGE_NAME=${CLIENT_IMAGE_NAME:-fl-client}
SERVER_IMAGE_NAME=${SERVER_IMAGE_NAME:-fl-server}
TAG=${TAG:-latest}

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }

echo ""
echo "=========================================="
echo "  Optimized Minikube FL Deployment"
echo "=========================================="
echo ""

# Check prerequisites
if ! command -v minikube &> /dev/null; then
    print_error "Minikube not found. Install with: brew install minikube"
    exit 1
fi

if ! command -v kubectl &> /dev/null; then
    print_error "kubectl not found. Install with: brew install kubectl"
    exit 1
fi

# Check if Minikube is running
if ! minikube status &> /dev/null; then
    print_error "Minikube is not running. Start it with: minikube start --cpus=4 --memory=8192"
    exit 1
fi

print_success "Prerequisites checked"

# Use Minikube's Docker daemon
print_status "Switching to Minikube Docker environment..."
eval $(minikube docker-env)

# Build images
print_status "Building Docker images..."
docker build -t ${CLIENT_IMAGE_NAME}:${TAG} -f docker/client/Dockerfile .
docker build -t ${SERVER_IMAGE_NAME}:${TAG} -f docker/server/Dockerfile .

print_success "Images built"

# Reset Docker environment
eval $(minikube docker-env -u)

# Create namespace
print_status "Creating namespace..."
kubectl create namespace ${NAMESPACE} 2>/dev/null || print_warning "Namespace already exists"

# Create Kaggle secret
if ! kubectl get secret kaggle-secret -n ${NAMESPACE} &>/dev/null; then
    print_status "Creating Kaggle secret..."
    if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
        print_error "Kaggle credentials not found at ~/.kaggle/kaggle.json"
        print_error "Set up with: mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
        exit 1
    fi
    kubectl create secret generic kaggle-secret \
      --from-file=kaggle.json=$HOME/.kaggle/kaggle.json \
      -n ${NAMESPACE}
fi

# Create ConfigMaps
print_status "Creating ConfigMaps..."
kubectl create configmap fl-server-config \
  --from-file=config/k8s-server.yaml \
  -n ${NAMESPACE} \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl create configmap fl-client-config \
  --from-file=config/k8s-client.yaml \
  -n ${NAMESPACE} \
  --dry-run=client -o yaml | kubectl apply -f -

# Apply storage (PVC for dataset cache)
print_status "Creating storage for dataset cache..."
kubectl apply -f k8s/storage/ -n ${NAMESPACE}

# Wait for PVC to be bound
print_status "Waiting for PVC to be ready..."
kubectl wait --for=condition=Bound pvc/dataset-cache-pvc -n ${NAMESPACE} --timeout=60s 2>/dev/null || print_warning "PVC may not be bound yet"

# Run preprocessing job (if not already completed)
print_status "Checking if dataset preprocessing is needed..."
if kubectl get job preprocess-dataset -n ${NAMESPACE} 2>/dev/null | grep -q "1/1"; then
    print_success "Dataset already preprocessed, skipping..."
else
    print_status "Running dataset preprocessing job (one-time, 3-5 minutes)..."
    kubectl delete job preprocess-dataset -n ${NAMESPACE} 2>/dev/null || true
    sleep 2
    kubectl apply -f k8s/jobs/preprocess-dataset.yaml -n ${NAMESPACE}
    
    print_status "Waiting for preprocessing to complete..."
    kubectl wait --for=condition=complete job/preprocess-dataset -n ${NAMESPACE} --timeout=600s
    
    if [ $? -eq 0 ]; then
        print_success "Dataset preprocessing completed!"
    else
        print_error "Preprocessing failed. Check logs: kubectl logs -n ${NAMESPACE} job/preprocess-dataset"
        exit 1
    fi
fi

# Deploy services
print_status "Deploying services..."
kubectl apply -f k8s/services/ -n ${NAMESPACE}

# Deploy server
print_status "Deploying FL server..."
kubectl apply -f k8s/deployments/fl-server.yaml -n ${NAMESPACE}
kubectl set image deployment/fl-server fl-server=${SERVER_IMAGE_NAME}:${TAG} -n ${NAMESPACE}

# Wait for server to be ready
print_status "Waiting for server to be ready..."
kubectl wait --for=condition=available deployment/fl-server -n ${NAMESPACE} --timeout=120s

# Deploy optimized clients (use Deployment for parallel startup)
print_status "Deploying optimized FL clients..."
kubectl apply -f k8s/deployments/fl-client-optimized.yaml -n ${NAMESPACE}
kubectl set image deployment/fl-clients-optimized fl-client=${CLIENT_IMAGE_NAME}:${TAG} -n ${NAMESPACE}

# Wait for clients
print_status "Waiting for clients to start..."
sleep 10

echo ""
print_success "✅ Optimized deployment complete!"
echo ""
echo "=========================================="
echo "  ⚡ Performance Optimizations Active"
echo "=========================================="
echo ""
echo "✅ Dataset downloaded ONCE (shared via PVC)"
echo "✅ Dataset preprocessed ONCE (cached)"
echo "✅ Clients start in PARALLEL (Deployment)"
echo "✅ Reduced memory per client (1GB vs 3GB)"
echo ""
echo "Expected time: 10-20 minutes (vs 120 min before)"
echo ""
echo "=========================================="
echo "  📊 Monitoring Commands"
echo "=========================================="
echo ""
echo "View pods:"
echo "  kubectl get pods -n ${NAMESPACE}"
echo ""
echo "View server logs:"
echo "  kubectl logs -f deployment/fl-server -n ${NAMESPACE}"
echo ""
echo "View client logs:"
echo "  kubectl logs -f deployment/fl-clients-optimized -n ${NAMESPACE}"
echo ""
echo "Scale clients:"
echo "  kubectl scale deployment fl-clients-optimized --replicas=30 -n ${NAMESPACE}"
echo ""
echo "Check status:"
echo "  kubectl get all -n ${NAMESPACE}"
echo ""
echo "=========================================="
echo ""

# Show current status
print_status "Current pod status:"
kubectl get pods -n ${NAMESPACE}
