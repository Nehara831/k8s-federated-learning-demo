#!/usr/bin/env bash
set -euo pipefail

NAMESPACE=${NAMESPACE:-federated-learning}
CLIENT_IMAGE_NAME=${CLIENT_IMAGE_NAME:-fl-client}
SERVER_IMAGE_NAME=${SERVER_IMAGE_NAME:-fl-server}
TAG=${TAG:-latest}

echo "[i] Ensuring namespace exists: ${NAMESPACE}"
kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

# Apply all manifests (adjust path if different)
if [ -d "k8s" ]; then
  echo "[i] Applying manifests from ./k8s"
  kubectl apply -n "${NAMESPACE}" -f k8s
else
  echo "[!] ./k8s directory not found. Skipping manifest apply."
fi

echo "[i] Setting images on deployments"
kubectl set image deployment/fl-clients fl-client=${CLIENT_IMAGE_NAME}:${TAG} -n "${NAMESPACE}"
kubectl set image deployment/fl-server  fl-server=${SERVER_IMAGE_NAME}:${TAG} -n "${NAMESPACE}"

echo "[i] Waiting for rollouts to complete"
kubectl rollout status deployment/fl-server  -n "${NAMESPACE}"
kubectl rollout status deployment/fl-clients -n "${NAMESPACE}"

echo
echo "[✓] Deploy complete."
echo "[i] Pods:"
kubectl get pods -n "${NAMESPACE}"

echo
echo "[i] Tail server logs:"
echo "kubectl logs deployment/fl-server -n ${NAMESPACE} --tail=100 -f"
echo "[i] Tail all client logs:"
echo "kubectl logs -l app=fl-client -n ${NAMESPACE} --tail=100 -f"