import subprocess
import time

def main():
    # 1. Define your list of maps
    maps_to_run = [
        "Town01", 
        "Town02", 
        "Town03", 
        "Town04",
        "Town05", 
        "Town10HD"
    ]
    
    # 2. Define your consistent parameters here
    frames_per_map = 200
    vehicles = 40
    walkers = 40

    print("🚀 Starting Automated CARLA Dataset Pipeline...")
    
    for current_map in maps_to_run:
        print(f"\n{'='*50}")
        print(f"   Starting Capture for Map: {current_map}")
        print(f"{'='*50}")
        
        # 3. Build the command exactly as you would type it in the terminal
        command = [
            "python", "run_dataset.py",
            "--map", current_map,
            "--frames", str(frames_per_map),
            "--vehicles", str(vehicles),
            "--walkers", str(walkers),
            # You can add flags here too, e.g., "--safe", "--car-lights-on"
        ]
        
        try:
            print(f"Executing: {' '.join(command)}")
            # subprocess.run will wait until run_dataset.py finishes before continuing
            subprocess.run(command, check=True)
            print(f"✅ Successfully finished {current_map}")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error occurred while processing {current_map}.")
            print(f"Details: {e}")
            print("Skipping to the next map...")
            
        # 4. Give the CARLA server a 5-second breather to clear memory 
        # before we bombard it with the next world-load command.
        print("Cooling down for 5 seconds...")
        time.sleep(5)

    print("\n🎉 Pipeline complete! All maps processed.")

if __name__ == "__main__":
    main()
