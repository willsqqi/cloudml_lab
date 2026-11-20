# Homework 4 Kubernetes MNIST

## Files

- `src/`: model, training, inference and web page.
- `Dockerfile.train`, `Dockerfile.inference`: container definitions.
- `k8s/`: storage, training Job, inference Deployment and Service.
- `cloudbuild.yaml`: build and push both images.
- `screenshots/`: screenshots from each run.
- `scripts/cleanup.sh`: remove cloud resources after saving results.

## Workflow

1. Build and push the images.
2. Run training and save the model on persistent storage.
3. Deploy inference and test predictions through the URL.
4. Save screenshots and results, clean up.