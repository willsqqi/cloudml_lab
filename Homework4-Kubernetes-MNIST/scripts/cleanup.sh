#!/usr/bin/env bash
# Run from the Homework 4 folder after exporting GKE evidence and the model.
set -euo pipefail
project_id="${1:?Usage: scripts/cleanup.sh PROJECT_ID RUN_DIR}"
run_dir="$(cd "${2:?Usage: scripts/cleanup.sh PROJECT_ID RUN_DIR}" && pwd)"
region=us-central1
cluster=homework4-mnist

# Do not remove the only copy of the cloud model or its training log.
test -s "$run_dir/model/mnist_cnn.pt"
test -s "$run_dir/evidence/training-gke.log"
gcloud container clusters get-credentials "$cluster" --region "$region" --project "$project_id"

pv=$(kubectl -n homework4 get pvc mnist-model -o jsonpath='{.spec.volumeName}')
volume=$(kubectl get pv "$pv" -o jsonpath='{.spec.csi.volumeHandle}')
disk_name="${volume##*/}"
service_ip=$(kubectl -n homework4 get service mnist-inference -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
if [[ "$volume" != "projects/$project_id/zones/"* || -z "$disk_name" || -z "$service_ip" ]]; then
    echo "Could not identify this assignment's disk and load balancer; stopping." >&2
    exit 1
fi
if [[ "$(kubectl get pv "$pv" -o jsonpath='{.spec.persistentVolumeReclaimPolicy}')" != Delete ]]; then
    echo "The volume does not have reclaim policy Delete; stopping." >&2
    exit 1
fi

# Deleting the namespace removes the Service, workloads and PVC first.
kubectl delete namespace homework4 --wait=true --timeout=10m

# Successful empty list responses establish removal; API errors stop the script.
for attempt in {1..30}; do
    rules=$(gcloud compute forwarding-rules list --project "$project_id" \
        --filter="IPAddress=$service_ip" --format='value(name)')
    disks=$(gcloud compute disks list --project "$project_id" \
        --filter="name=$disk_name" --format='value(name)')
    if [[ -z "$rules" && -z "$disks" ]]; then
        break
    fi
    if [[ "$attempt" == 30 ]]; then
        echo "Load balancer or disk remains; the cluster has been left in place." >&2
        exit 1
    fi
    sleep 10
done

gcloud container clusters delete "$cluster" --region "$region" --project "$project_id" --quiet
# This repository is dedicated to Homework 4; deleting it removes both images.
gcloud artifacts repositories delete homework4 --location "$region" --project "$project_id" --quiet
for identity in build node; do
    role=roles/cloudbuild.builds.builder
    if [[ "$identity" == node ]]; then
        role=roles/container.defaultNodeServiceAccount
    fi
    member="serviceAccount:homework4-$identity@$project_id.iam.gserviceaccount.com"
    gcloud projects remove-iam-policy-binding "$project_id" --member="$member" \
        --role="$role" --condition=None --quiet --format=none
    gcloud iam service-accounts delete "homework4-$identity@$project_id.iam.gserviceaccount.com" \
        --project "$project_id" --quiet
done
echo "Homework 4 namespace, load balancer, disk, cluster, images and service accounts removed."
