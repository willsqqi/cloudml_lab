"""Download MNIST before running the application, including during image builds."""

import argparse
from pathlib import Path

from torchvision.datasets import MNIST
from torchvision.datasets.utils import download_and_extract_archive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()
    raw_dir = Path(args.data_dir) / "MNIST" / "raw"

    # MNIST.download() fetches both splits even when train=False.
    for filename, checksum in MNIST.resources:
        if args.test_only and not filename.startswith("t10k-"):
            continue
        download_and_extract_archive(
            f"https://ossci-datasets.s3.amazonaws.com/mnist/{filename}",
            str(raw_dir),
            md5=checksum,
            remove_finished=True,
        )


if __name__ == "__main__":
    main()
