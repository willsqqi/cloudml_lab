"""GPU Chase ... Find a GPU in Google Cloud

PSEUDOCODE
    SET project, selected GPU (T4), VM name, and region filter below
    CREATE timestamped CSV and log file paths
    GET regions, zones, and GPU types from Google Cloud

    PASS 1 - Check availability
        FOR each region matching the filter:
            FOR each zone in that region:
                CHECK whether the selected GPU is listed
                ADD availability and initial allocation status to the table
                MARK non-operational zones as skipped
        SAVE the table to CSV

    ASK for RUN before starting paid tests; otherwise STOP

    PASS 2 - Attempt allocation
        FOR each operational zone that lists the selected GPU:
            CHECK that no previous test VM or boot disk remains
            ATTEMPT to create a VM with one selected GPU
            LOG the command, exit code, output, and errors
            RECORD allocated = Yes if creation succeeds, otherwise No
            RECORD the failure reason if creation fails
            SAVE the updated CSV
            DELETE the test VM and any remaining boot disk
            VERIFY cleanup; STOP if deletion or verification fails

    PRINT the CSV and log locations

"""

import csv
from datetime import datetime
from pathlib import Path
import json
import subprocess

PROJECT = "cloudml-509216"
GPU = "nvidia-tesla-t4"
NAME = "gpu-chase-test"
REGION_PREFIX = "us-"  # "us-" = US regions

LOG = None


def run_command(command, check=False):
    result = subprocess.run(command, capture_output=True, text=True)
    with LOG.open("a") as file:
        file.write(f"\n[{datetime.now().astimezone().isoformat()}] {' '.join(command)}\n")
        file.write(f"Exit code: {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\n")
    if check:
        result.check_returncode()
    return result


def read_cloud(*args):
    # read cloud data.
    result = run_command(
        ["gcloud", "compute", *args, f"--project={PROJECT}", "--format=json"], check=True)
    return json.loads(result.stdout)


def check_cleanup():
    # Check for leftover.
    for resource in ["instances", "disks"]:
        if read_cloud(resource, "list", f"--filter=name={NAME}"):
            raise SystemExit(f"STOP: {NAME} still exists in {resource}. Delete it manually first.")


def main():
    global LOG
    output = Path(__file__).with_name("results-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".csv")
    LOG = output.with_suffix(".log")
    LOG.touch()
    zones = read_cloud("zones", "list")
    gpu_types = read_cloud("accelerator-types", "list")
    rows = []
    for region in read_cloud("regions", "list"):
        if not region["name"].startswith(REGION_PREFIX):
            continue
        for zone in zones:
            if zone["region"].split("/")[-1] != region["name"]:
                continue
            offered = any(g["name"] == GPU and g["zone"].split("/")[-1] == zone["name"]
                          for g in gpu_types)
            reason = "Not attempted" if offered else "T4 not offered; skipped"
            rows.append(dict(region=region["name"], zone=zone["name"], gpu=GPU,
                             available="Yes" if offered else "No", attempted="No",
                             allocated="Not tested" if offered else "No", reason=reason))
            if zone["status"] != "UP":
                rows[-1]["reason"] = "Zone not UP; skipped"
            print(region["name"], zone["name"], "T4 listed:", rows[-1]["available"])

    def save():
        with output.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["region", "zone", "gpu", "available", "attempted", "allocated", "reason"])
            writer.writeheader()
            writer.writerows(rows)

    save()
    if input("Type RUN to test all eligible zones, or Enter for discovery only: ") != "RUN":
        return
    for row in rows:
        if row["reason"] != "Not attempted":
            continue
        check_cleanup()
        zone = row["zone"]
        print(f"\nTesting {zone} with one {GPU} ...", flush=True)
        result = run_command([
            "gcloud", "compute", "instances", "create", NAME, f"--project={PROJECT}",
            f"--zone={zone}", "--machine-type=n1-standard-1",
            f"--accelerator=type={GPU},count=1", "--maintenance-policy=TERMINATE",
            "--image-family=debian-12", "--image-project=debian-cloud",
            "--boot-disk-size=10GB", "--boot-disk-type=pd-standard",
            "--boot-disk-auto-delete", "--network=default", "--no-address", "--quiet"])
        print(result.stdout, result.stderr)
        success = result.returncode == 0
        row["attempted"] = "Yes"
        row["allocated"] = "Yes" if success else "No"
        row["reason"] = "Created" if success else result.stderr.strip() or result.stdout.strip()
        save()
        print("GPU allocated:", row["allocated"], "— result saved.")
        # Delete VM, then leftover disk.
        for resource in ["instances", "disks"]:
            if read_cloud(resource, "list", f"--filter=name={NAME} AND zone:{zone}"):
                run_command(["gcloud", "compute", resource, "delete", NAME,
                                f"--project={PROJECT}", f"--zone={zone}", "--quiet"], check=True)
        check_cleanup()
        with LOG.open("a") as file:
            file.write(f"{zone}: cleanup verified; test VM and disk absent.\n")
    print(f"Finished. Table: {output}\nLog: {LOG}")


if __name__ == "__main__":
    main()
