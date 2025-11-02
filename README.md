# Kubernetes Federated Learning Demo

A production-ready Federated Learning system deployed on Kubernetes, using Flower framework and 5G Network Intrusion Detection Dataset (5G-NIDD).

## 🚀 Features

- ✅ **Kubernetes-native deployment** with StatefulSets
- ✅ **Automatic dataset loading** from Kaggle
- ✅ **Multi-client federation** with data partitioning
- ✅ **FedAvg aggregation** with configurable rounds
- ✅ **Real-world dataset**: 1.2M rows from 5G-NIDD
- ✅ **Attack simulation** support (poisoning, backdoor)

## 📊 Results

| Round | Loss | Improvement |
|-------|------|-------------|
| 1 | 2.0699 | - |
| 2 | 1.8123 | ↓ 12.4% |
| 3 | 1.4253 | ↓ 21.4% |
| 4 | 1.0227 | ↓ 28.3% |
| 5 | **0.7495** | ↓ 26.7% |

**Total improvement: 63.8% reduction in loss across 5 rounds!**

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│           Kubernetes Cluster                │
│                                             │
│  ┌──────────────┐                          │
│  │  FL Server   │ ← Aggregates parameters  │
│  │   (1 pod)    │                          │
│  └──────────────┘                          │
│         ↑                                   │
│         │ gRPC                              │
│    ┌────┴────┬────────┬────────┐          │
│    ↓         ↓        ↓        ↓          │
│ ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐       │
│ │ C-0 │  │ C-1 │  │ C-2 │  │ C-N │       │
│ └─────┘  └─────┘  └─────┘  └─────┘       │
│  StatefulSet (N clients)                   │
│                                             │
└─────────────────────────────────────────────┘
```

## 📋 Prerequisites

- Kubernetes cluster (Minikube or cloud)
- Docker
- Python 3.9+
- Kaggle API credentials

## 🔧 Installation

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/k8s-fl-demo.git
cd k8s-fl-demo
```

### 2. Set Up Kaggle Credentials

```bash
# Download kaggle.json from https://www.kaggle.com/settings
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Build Docker Images

```bash
# Use Minikube's Docker daemon
eval $(minikube docker-env)

# Build images
docker build -t fl-server:v10 -f docker/server/Dockerfile .
docker build -t fl-client:v10 -f docker/client/Dockerfile .
```

### 4. Deploy to Kubernetes

```bash
# Create namespace
kubectl create namespace federated-learning

# Create Kaggle secret
kubectl create secret generic kaggle-secret \
  --from-file=kaggle.json=/home/$USER/.kaggle/kaggle.json \
  -n federated-learning

# Create ConfigMaps
kubectl create configmap fl-server-config \
  --from-file=config/k8s-server.yaml \
  -n federated-learning

kubectl create configmap fl-client-config \
  --from-file=config/k8s-client.yaml \
  -n federated-learning

# Deploy storage
kubectl apply -f k8s/storage/

# Deploy server and clients
kubectl apply -f k8s/deployments/fl-server.yaml
kubectl apply -f k8s/deployments/fl-client-statefulset.yaml
```

## 📊 Monitoring

```bash
# Watch pods
kubectl get pods -n federated-learning -w

# View server logs
kubectl logs -f deployment/fl-server -n federated-learning

# View client logs
kubectl logs -f fl-clients-0 -n federated-learning
kubectl logs -f fl-clients-1 -n federated-learning
kubectl logs -f fl-clients-2 -n federated-learning
```

## ⚙️ Configuration

### Server Config (`config/k8s-server.yaml`)

```yaml
dataset:
  type: "5gnidd"
  test_size: 0.2
  seed: 42

num_rounds: 5
num_clients: 3
batch_size: 256
num_classes: 9

server:
  port: 8080
  min_fit_clients: 3
  min_evaluate_clients: 3
  min_available_clients: 3
```

### Client Config (`config/k8s-client.yaml`)

```yaml
dataset:
  type: "5gnidd"

num_clients: 3
batch_size: 256

config_fit:
  lr: 0.01
  momentum: 0.9
  local_epochs: 1
```

## 🔬 Scaling

Change number of clients:

```bash
# Edit fl-client-statefulset.yaml
replicas: 5  # Change from 3 to 5

# Update configs to match
# config/k8s-server.yaml: num_clients: 5, min_fit_clients: 5
# config/k8s-client.yaml: num_clients: 5

# Redeploy
kubectl apply -f k8s/deployments/fl-client-statefulset.yaml
```

## 🛡️ Attack Simulation

Enable poisoning attacks:

```yaml
# config/k8s-client.yaml
attack:
  enabled: true
  attack_type: "label_flipping"
  malicious_ratio: 0.3  # 30% of clients are malicious
```

## 📁 Project Structure

```
k8s-fl-demo/
├── config/              # Configuration files
├── docker/              # Dockerfiles
├── k8s/                 # Kubernetes manifests
│   ├── deployments/
│   └── storage/
├── src/
│   ├── client/          # FL client implementation
│   ├── server/          # FL server implementation
│   └── shared/          # Shared utilities (dataset, models)
└── README.md
```

## 📝 Dataset

This project uses the **5G-NIDD** (5G Network Intrusion Detection Dataset):
- **Source**: [Kaggle](https://www.kaggle.com/datasets/humera11/5g-nidd-dataset)
- **Size**: 1.2M network traffic records
- **Classes**: 9 attack types
- **Features**: 48 network flow characteristics

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a PR.

## 📄 License

MIT License

## 🙏 Acknowledgments

- [Flower Framework](https://flower.dev/)
- [5G-NIDD Dataset](https://www.kaggle.com/datasets/humera11/5g-nidd-dataset)
- Kubernetes community

## 📧 Contact

Your Name - your.email@example.com

Project Link: https://github.com/YOUR_USERNAME/k8s-fl-demo