"""
Runs the CARLA data capture pipeline sequentially across multiple maps.

Usage:
    python batch_generate.py
"""
import subprocess
import time


MAPS = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD"]
FRAMES_PER_MAP = 200
VEHICLES = 40
WALKERS = 40


def main():
    print("Starting Automated CARLA Dataset Pipeline...")

    for current_map in MAPS:
        print(f"\n{'='*50}")
        print(f"   Map: {current_map}")
        print(f"{'='*50}")

        command = [
            "python", "run_dataset.py",
            "--map", current_map,
            "--frames", str(FRAMES_PER_MAP),
            "--vehicles", str(VEHICLES),
            "--walkers", str(WALKERS),
        ]

        try:
            print(f"Running: {' '.join(command)}")
            subprocess.run(command, check=True)
            print(f"Finished {current_map}")
        except subprocess.CalledProcessError as e:
            print(f"Error on {current_map}: {e} — skipping.")

        # Brief cooldown before loading the next map to let CARLA free memory.
        time.sleep(5)

    print("\nPipeline complete. All maps processed.")


if __name__ == "__main__":
    main()
