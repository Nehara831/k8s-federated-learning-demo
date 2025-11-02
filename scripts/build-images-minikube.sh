#!/usr/bin/env bash
set -euo pipefail

NAMESPACE=${NAMESPACE:-federated-learning}
CLIENT_IMAGE_NAME=${CLIENT_IMAGE_NAME:-fl-client}
SERVER_IMAGE_NAME=${SERVER_IMAGE_NAME:-fl-server}
TAG=${TAG:-$(date +%Y%m%d%H%M%S)}

echo "[i] Using namespace: ${NAMESPACE}"
echo "[i] Building images with Minikube Docker daemon"
eval "$(minikube docker-env)"

echo "[i] Building client image: ${CLIENT_IMAGE_NAME}:${TAG}"
docker build -t ${CLIENT_IMAGE_NAME}:${TAG} -f docker/client/Dockerfile .

echo "[i] Building server image: ${SERVER_IMAGE_NAME}:${TAG}"
docker build -t ${SERVER_IMAGE_NAME}:${TAG} -f docker/server/Dockerfile .

echo "[i] Images built:"
docker images | grep -E "${CLIENT_IMAGE_NAME}|${SERVER_IMAGE_NAME}" | head

# Optional: also tag as :latest
docker tag ${CLIENT_IMAGE_NAME}:${TAG} ${CLIENT_IMAGE_NAME}:latest
docker tag ${SERVER_IMAGE_NAME}:${TAG} ${SERVER_IMAGE_NAME}:latest

echo "[i] Resetting local Docker env"
eval "$(minikube docker-env -u)"

echo
echo "[✓] Done. Export TAG and run deploy script:"
echo "export TAG=${TAG} && scripts/deploy-minikube.sh"