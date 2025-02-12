import subprocess

# Run the first script
subprocess.run(["py", "./random clips/vidtheif.py"], check=True)

# Run the second script
subprocess.run(["py", "hardcoreparkour.py"], check=True)

# Run the third script after the first one finishes
subprocess.run(["py", "./next steps/mushroom.py"], check=True)