# MLOps PyTorch Pipeline

## Project overview

This repository implements a small end-to-end MLOps workflow for a CIFAR-10 image classifier trained in PyTorch and deployed with Docker and Kubernetes. The project covers model training, artifact persistence, containerized serving, Kubernetes manifests, readiness/liveness checks, and end-to-end validation.

## Architecture

```text
+--------------------+      +--------------------+      +----------------------+
| Local developer    | ---> | Docker build/test  | ---> | Kubernetes cluster    |
| or Git repo        |      | training + serving |      | ml-training namespace |
+--------------------+      +--------------------+      +----------+-----------+
                                                                  |
                                                                  v
                                           +------------------------------+
                                           | Training Job                 |
                                           | mlops-training               |
                                           | reads data + writes model    |
                                           +------------------------------+
                                                                  |
                                                                  v
                                           +------------------------------+
                                           | Shared PVC                   |
                                           | checkpoints-pvc              |
                                           | checkpoints + data           |
                                           +------------------------------+
                                                                  |
                                                                  v
                                           +------------------------------+
                                           | Serving Deployment           |
                                           | model-serving                |
                                           | /health + /predict API       |
                                           +------------------------------+
                                                                  |
                                                                  v
                                           +------------------------------+
                                           | Service                      |
                                           | model-serving                |
                                           +------------------------------+
```

## Repository structure

```text
.
├── checkpoints/
│   └── classifier_v1.pt
├── configs/
│   └── training_config.yaml
├── data/
│   ├── cifar-10-batches-py/
│   └── test_image.png
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.serve
├── k8s/
│   ├── configmap.yaml
│   ├── hpa.yaml
│   ├── pvc.yaml
│   ├── serving-deployment.yaml
│   ├── serving-service.yaml
│   ├── namespace.yaml
│   └── training-job.yaml
├── requirements/
│   ├── serve.txt
│   └── train.txt
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── serve.py
│   └── train.py
├── .gitignore
├── README.md
└── .venv/
```

## Docker training

Build the training image and run it locally when needed:

```bash
docker build -f docker/Dockerfile.train -t mlops-pytorch-train:dev .
docker run --rm -it \
  -e MODEL_PATH=checkpoints/classifier_v1.pt \
  -v ${PWD}/data:/app/data \
  -v ${PWD}/checkpoints:/app/checkpoints \
  mlops-pytorch-train:dev
```

The training pipeline loads CIFAR-10 data, trains a small ResNet-based classifier, and writes the checkpoint to the configured checkpoint directory.

## Docker serving

Build and run the serving image:

```bash
docker build -f docker/Dockerfile.serve -t mlops-pytorch-serve:dev .
docker run --rm -p 8080:8080 mlops-pytorch-serve:dev
```

The service exposes:

- `GET /health`
- `POST /predict`

## Kubernetes setup

The Kubernetes manifests in the `k8s/` folder create the namespace, ConfigMap, persistent volume claim, training Job, inference Deployment, service, and autoscaler.

### Apply manifests

```bash
kubectl apply -f k8s/
```

For client-side validation only:

```bash
kubectl apply --dry-run=client -f k8s/
```

## Checking the training Job

```bash
kubectl get job -n ml-training
kubectl describe job mlops-training -n ml-training
kubectl logs job/mlops-training -n ml-training --tail=100
```

Successful completion is indicated by:

```bash
kubectl get job -n ml-training
# STATUS: Complete
```

The trained model checkpoint should exist in the shared PVC under `/app/checkpoints`.

## Checking the serving Deployment

```bash
kubectl get deployment -n ml-training
kubectl describe deployment model-serving -n ml-training
kubectl get pods -n ml-training -l app=model-serving
```

The deployment should report a ready replica and remain healthy according to the configured readiness and liveness probes.

## Testing `/health`

```bash
kubectl -n ml-training port-forward svc/model-serving 8080:80 --address 127.0.0.1
curl http://127.0.0.1:8080/health
```

Expected response:

```json
{"status":"ok"}
```

## Testing `/predict`

Use the existing test image in `data/test_image.png`:

```bash
curl -X POST "http://127.0.0.1:8080/predict" \
  -F "file=@data/test_image.png"
```

The endpoint returns a predicted class id, label, and probability vector.

## Checking the HPA

```bash
kubectl get hpa -n ml-training
kubectl describe hpa -n ml-training
```

The project includes a minimal `autoscaling/v2` HPA for `model-serving` with:

- `minReplicas: 1`
- `maxReplicas: 3`
- CPU target utilization: `70%`

### Metrics availability note

In this Docker Desktop / kind environment, the cluster does not currently expose CPU metrics to the HPA because `metrics-server` is present but cannot scrape the kubelet endpoint successfully. The system logs show a TLS certificate validation failure when contacting the node's `https://<node-ip>:10250/metrics/resource` endpoint. This is why the HPA reports `cpu: <unknown>/70%` instead of an actual utilization value.

## Cleanup commands

```bash
kubectl delete -f k8s/
```

Or remove individual objects:

```bash
kubectl delete namespace ml-training
kubectl delete deployment model-serving -n ml-training
kubectl delete service model-serving -n ml-training
kubectl delete job mlops-training -n ml-training
```

## End-to-end workflow

```text
1. Train model with Docker or Kubernetes Job.
2. Confirm the checkpoint is saved to the PVC.
3. Apply the Kubernetes manifests.
4. Wait for the training job to complete successfully.
5. Check that the inference deployment is ready.
6. Verify the service has an endpoint.
7. Test /health and /predict through the service.
8. Inspect the HPA and confirm whether metrics are available.
9. Remove resources when done.
```

This repository is intended as a simple, assignment-appropriate MLOps example and keeps the application code and model behavior unchanged while validating the deployment workflow end to end.
