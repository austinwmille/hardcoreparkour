This script generates Minecraft parkour 'brainrot' videos.

I use a Python 3.9 environment for reasons I cannot remember, but the package requirements are listed.

Running requires a folder containing at least one minecraft parkour video.
It also requires a separate folder containing movies, videos, interviews, or other video content which we want to turn into brainrot.

The paths for these folders are hardcoded in the script for my computer. You must change these to your own, actual paths.

The output folder is /'random clips'.

As of 02/07/2025, it works thusly:

	I have a folder of maybe four or five (at least 1) 30 minute long Minecraft parkour videos.
	
	I have a folder of an arbitrary number (at least 1) of longform, provacative interview/debate style videos.
	
	This script chooses a random video from the longform folder.
	It then it chooses a random 10 minute segment from its duration (unless the chosen video is less than 10 minutes, then it uses the whole thing).
	
	It uses openCV to choose a decent screenshot, applies a border and soft filter using ffmpeg.
	This will be the thumbnail, but hasn't been tested for optimization much. 

	It then picks a random video from the Minecraft parkour folder, and clips a segment from it which is equal in length to the above clip (currently hardcoded as 10 minutes, unless the video is shorter).

	The two equal length clips (one from the longform video, one from the parkour video) are then stacked to make a single video file.

	The final (usually 10 minute long) video is then cut into 2 minute long segments.

	