import subprocess

#step one, download a video
# Run the first script
subprocess.run(["py", "./vidthief.py"], check=True) 

#step 2, combine with random segment of minecraft parkour video
# Run the second script
subprocess.run(["py", "../hardcoreparkour.py"], check=True)

#call in LLMs and APIs to transcribe the video, lay subs, and attach metadata and upload to Youtube
# Run the third script 
subprocess.run(["py", "../next steps/mushroom.py"], check=True)