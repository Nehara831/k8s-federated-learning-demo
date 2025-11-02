#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <number_of_clients>"
    exit 1
fi

NUM_CLIENTS=$1

echo "Scaling clients to $NUM_CLIENTS replicas..."
kubectl scale deployment fl-clients --replicas=$NUM_CLIENTS -n federated-learning

echo "Scaled to $NUM_CLIENTS clients"
kubectl get pods -n federated-learning -l app=fl-client