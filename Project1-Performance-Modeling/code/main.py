import argparse
import copy
import json
import platform
import statistics
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torchvision
from torchvision import datasets, models, transforms

# Keep these settings identical on both GPUs.
BATCH_SIZE = 32
SUBSET_SIZE = 128
WARMUP_STEPS = 10
TIMED_STEPS = 50
NUM_REPEATS = 3
RANDOM_SEED = 0


def load_batches(data_folder):
    """Choose the same images each time and prepare them before timing."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(data_folder, transform=transform)
    if len(dataset) < SUBSET_SIZE or len(dataset.classes) > 1000:
        raise ValueError("Use at least 128 images and at most 1,000 class folders.")
    # The fixed seed selects the same images in the same order on both GPUs.
    random_generator = torch.Generator().manual_seed(RANDOM_SEED)
    image_indices = torch.randperm(len(dataset), generator=random_generator)[:SUBSET_SIZE].tolist()
    image_batches, selected_images = [], []
    for batch_start in range(0, SUBSET_SIZE, BATCH_SIZE):
        batch_indices = image_indices[batch_start:batch_start + BATCH_SIZE]
        images, labels = zip(*(dataset[image_index] for image_index in batch_indices))
        image_batches.append((torch.stack(images), torch.tensor(labels)))
        selected_images.extend(
            {"image": str(Path(dataset.samples[image_index][0]).relative_to(data_folder)),
             "label": dataset.samples[image_index][1]}
            for image_index in batch_indices
        )
    return image_batches, selected_images


# process one batch of images and update the model weights; main calls this repeatedly
def train_step(model, optimizer, image_batch, profile=False):
    """Make predictions, calculate gradients, and update the model once."""
    images, labels = image_batch
    optimizer.zero_grad(set_to_none=True)
    # NVTX names tell Nsight which kernels belong to each phase.
    with torch.cuda.nvtx.range("forward") if profile else nullcontext():
        predictions = model(images)
    loss = torch.nn.functional.cross_entropy(predictions, labels)
    with torch.cuda.nvtx.range("backward") if profile else nullcontext():
        loss.backward()
    optimizer.step()
    return loss.detach()


def main():
    """Prepare the experiment, run the measurements, and save the results."""
    # read the run options
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="ImageNet class-folder root")
    parser.add_argument("--model", choices=["resnet50", "resnet101", "resnet152"], default="resnet50")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--profile", action="store_true", help="Mark one warmed-up step; do not time it")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("This experiment requires an NVIDIA CUDA GPU.")

    torch.manual_seed(RANDOM_SEED)
    # FP32 tensors alone do not disable TF32 convolution or matrix multiplication.
    if hasattr(torch.backends, "fp32_precision"):
        torch.backends.fp32_precision = "ieee"
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True

    # load images onto the GPU before timing, so disk reads and copies are excluded.
    image_batches, selected_images = load_batches(args.data)
    image_batches = [(images.cuda(), labels.cuda()) for images, labels in image_batches]
    model = models.get_model(args.model, weights=None)
    initial_model_state = copy.deepcopy(model.state_dict())  # Keep the reset copy on CPU.
    model = model.cuda().train()
    repeat_measurements = []
    # repeat the experiment
    for repeat_index in range(1 if args.profile else NUM_REPEATS):
        # restore the starting weights
        model.load_state_dict(initial_model_state)
        # Starting a new optimizer resets its momentum for each repetition.
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, foreach=False)
        # warm up before measuring
        for step in range(WARMUP_STEPS):
            train_step(model, optimizer, image_batches[step % len(image_batches)])
        torch.cuda.synchronize()

        if args.profile:
            # mark one step for profiling
            loss = train_step(model, optimizer, image_batches[0], profile=True)
            torch.cuda.synchronize()
        else:
            torch.cuda.reset_peak_memory_stats()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            # time fifty training steps
            start_event.record()
            for step in range(TIMED_STEPS):
                loss = train_step(model, optimizer, image_batches[step % len(image_batches)])
            end_event.record()
            end_event.synchronize()  # Wait for the GPU before reading its elapsed time.
            total_time_ms = start_event.elapsed_time(end_event)
            # record this repetition
            repeat_measurements.append({
                "repeat": repeat_index + 1,
                "step_ms": total_time_ms / TIMED_STEPS,
                "images_per_second": BATCH_SIZE * TIMED_STEPS * 1000 / total_time_ms,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            })
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite loss; do not use this run's measurements.")

    # collect settings and measurements
    result = {
        "model": args.model, "gpu": torch.cuda.get_device_name(),
        "gpu_memory_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "python": platform.python_version(), "command": sys.argv,
        "precision": "FP32 (TF32 disabled)", "batch_size": BATCH_SIZE,
        "subset_size": SUBSET_SIZE, "seed": RANDOM_SEED, "warmup_steps": WARMUP_STEPS,
        "timed_steps": 0 if args.profile else TIMED_STEPS,
        "optimizer": "SGD, lr=0.01, momentum=0.9, foreach=False",
        "mode": "profile" if args.profile else "timing", "images": selected_images,
        "measurements": repeat_measurements,
    }
    # summarize the repetitions
    if repeat_measurements:
        result["summary"] = {}
        for metric in ["step_ms", "images_per_second", "peak_allocated_mib"]:
            values = [row[metric] for row in repeat_measurements]
            result["summary"][metric] = {
                "median": statistics.median(values), "min": min(values), "max": max(values),
            }
        print(json.dumps(result["summary"], indent=2))
    # save the results as json
    args.output.mkdir(parents=True, exist_ok=True)
    suffix = "-profile" if args.profile else ""
    output_path = args.output / f"{args.model}{suffix}.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
