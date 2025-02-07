import os
import random
import subprocess
import sys
import tempfile
import datetime
# from concurrent.futures import ThreadPoolExecutor   #this is for future multithreading logic

# Configuration
MOVIE_FOLDER = "C:/Users/austi/Desktop/ctr+X/processmesempai"
PARKOUR_FOLDER = "C:/Users/austi/Desktop/ctr+X/extrascripts/stock vids/minecraft parkour"
OUTPUT_FOLDER = "C:/Users/austi/Desktop/hardcoreparkour/random clips"
CLIP_DURATION = 600  # 10 minutes in seconds

def find_video_files(folder):
    """Return a list of video files in the given folder"""
    video_extensions = ('.mp4', '.mov', '.mkv')
    video_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(video_extensions):
                video_files.append(os.path.join(root, file))
    return video_files

def pick_random_file(folder):
    """Pick a random video file from the specified folder"""
    files = find_video_files(folder)
    if not files:
        print(f"No video files found in {folder}")
        sys.exit(1)
    return random.choice(files)

def get_duration(file_path):
    """Get duration of video file in seconds using ffprobe"""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'format=duration',
        '-of', 'csv=p=0',
        file_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        return int(duration)
    except subprocess.CalledProcessError as e:
        print(f"Error getting duration for {file_path}: {e}")
        sys.exit(1)

def pick_random_start(total_duration):
    """Pick a random start time for the clip"""
    max_start = total_duration - CLIP_DURATION
    if max_start <= 0:
        return 0
    return random.randint(0, max_start)

import cv2
import numpy as np

def extract_keyframes(video_file, output_folder):
    """Extracts keyframes from a video and saves them in a temporary folder."""
    keyframes_folder = os.path.join(output_folder, "keyframes")
    os.makedirs(keyframes_folder, exist_ok=True)
    
    cmd = [
        'ffmpeg', '-y',
        '-i', video_file,
        '-vf', "select='eq(pict_type\\,I)',scale=1280:720",  # Extracts keyframes only
        '-vsync', 'vfr',
        os.path.join(keyframes_folder, "frame_%04d.jpg")
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error extracting keyframes: {e}")
        return None

    return keyframes_folder

def choose_best_thumbnail(keyframes_folder):
    """Selects the best thumbnail from extracted keyframes using brightness and sharpness."""
    best_frame = None
    best_score = 0

    for frame_file in sorted(os.listdir(keyframes_folder)):  # Sort ensures order
        frame_path = os.path.join(keyframes_folder, frame_file)
        image = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        # Calculate sharpness using variance of the Laplacian
        sharpness = cv2.Laplacian(image, cv2.CV_64F).var()

        # Calculate brightness as the mean pixel intensity
        brightness = np.mean(image)

        # Score = Sharpness + (Brightness * Weight)
        score = sharpness + (brightness * 1.5)

        if score > best_score:
            best_score = score
            best_frame = frame_path

    return best_frame

def detect_faces_in_video(video_path):
    """Detect faces in a sample of frames from a video to determine the best crop area."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(video_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_frames = [int(frame_count * r) for r in [0.2, 0.4, 0.6, 0.8]]  # Sample 4 points in video

    detected_faces = []

    for frame_pos in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)  # Move to frame position
        ret, frame = cap.read()

        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            detected_faces.append((x, y, w, h))

    cap.release()

    if not detected_faces:
        return None  # No faces detected

    # Calculate average face position
    avg_x = int(sum([f[0] for f in detected_faces]) / len(detected_faces))
    avg_y = int(sum([f[1] for f in detected_faces]) / len(detected_faces))
    avg_w = int(sum([f[2] for f in detected_faces]) / len(detected_faces))
    avg_h = int(sum([f[3] for f in detected_faces]) / len(detected_faces))

    return avg_x, avg_y, avg_w, avg_h

def crop_video_based_on_faces(input_video, output_folder):
    """Crop the video to center around detected faces."""
    face_coords = detect_faces_in_video(input_video)

    if not face_coords:
        print("No faces detected, using default center crop.")
        crop_x = "(in_w-1080)/2"
        crop_y = "(in_h-1920)/2"
    else:
        x, y, w, h = face_coords
        crop_x = max(x - 100, 0)  # Offset left
        crop_y = max(y - 200, 0)  # Offset top
        crop_w = min(w + 200, 1080)  # Expand width
        crop_h = min(h + 300, 1920)  # Expand height

    output_cropped = os.path.join(output_folder, os.path.basename(input_video).replace(".mp4", "_cropped.mp4"))

    cmd = [
        "ffmpeg", "-y", "-i", input_video,
        "-vf", f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-c:a", "copy",
        output_cropped
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"Cropped video saved: {output_cropped}")
    except subprocess.CalledProcessError as e:
        print(f"Error cropping video: {e}")
        return None

    return output_cropped

import shutil  # Imported some things mid script to see where i used them

def generate_thumbnail(movie_file, output_folder):
    """Extracts keyframes, selects the best one as a thumbnail, and applies vignette + border."""
    keyframes_folder = extract_keyframes(movie_file, output_folder)

    if not keyframes_folder:
        print("No keyframes found, using default middle frame.")
        return extract_middle_frame(movie_file, output_folder)  # Fallback

    best_frame = choose_best_thumbnail(keyframes_folder)

    if best_frame:
        final_thumbnail = os.path.join(output_folder, os.path.basename(movie_file).replace(".mp4", ".jpg"))
        
        # Apply vignette + border effect
        enhanced_thumbnail = final_thumbnail.replace(".jpg", "_enhanced.jpg")
        
        cmd = [
            "ffmpeg", "-y", "-i", best_frame,
            "-vf", "vignette=PI/4, pad=width=iw+60:height=ih+60:x=30:y=30:color=black",
            enhanced_thumbnail
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"Thumbnail saved: {enhanced_thumbnail}")
            final_thumbnail = enhanced_thumbnail  # Update the final thumbnail
        except subprocess.CalledProcessError as e:
            print(f"Error applying vignette/border: {e}")

    else:
        print("No suitable keyframes found, using default middle frame.")
        final_thumbnail = extract_middle_frame(movie_file, output_folder)  # Fallback

    # Cleanup extracted keyframes
    shutil.rmtree(keyframes_folder, ignore_errors=True)

    return final_thumbnail

def extract_middle_frame(video_file, output_folder):
    """Extracts a single frame from the middle of the video as a fallback thumbnail."""
    middle_time = get_duration(video_file) // 2
    thumbnail_path = os.path.join(output_folder, os.path.basename(video_file).replace(".mp4", "_fallback.jpg"))

    cmd = [
        'ffmpeg', '-y',
        '-i', video_file,
        '-ss', str(middle_time),
        '-vframes', '1',
        '-q:v', '2',
        thumbnail_path
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Fallback thumbnail saved: {thumbnail_path}")
        return thumbnail_path
    except subprocess.CalledProcessError as e:
        print(f"Error extracting fallback thumbnail: {e}")
        return None

def main():
    output_folder = "C:/Users/austi/Desktop/hardcoreparkour/random clips"
    
    for i in range(10):  # Run X times

        # Create temporary files
        temp_dir = tempfile.gettempdir()
        movie_clip = os.path.join(temp_dir, f"movie_clip_{os.getpid()}_{i}.mp4")
        parkour_clip = os.path.join(temp_dir, f"parkour_clip_{os.getpid()}_{i}.mp4")

        try:
            # 1. Pick random files
            movie_file = pick_random_file(MOVIE_FOLDER)
            parkour_file = pick_random_file(PARKOUR_FOLDER)

            # 2. Get durations
            movie_duration = get_duration(movie_file)
            parkour_duration = get_duration(parkour_file)

            # 3. Pick start times
            movie_start = pick_random_start(movie_duration)
            parkour_start = pick_random_start(parkour_duration)

            # 4. Extract clips
            ffmpeg_extract(movie_file, movie_start, movie_clip)
            ffmpeg_extract(parkour_file, parkour_start, parkour_clip)

            # 5. Stack videos
            output_file = stack_videos(movie_clip, parkour_clip, output_folder, movie_file)

            # 6. Generate thumbnail
            # Remove old thumbnails before generating new ones
            # Remove old thumbnail for the current video before generating a new one
            thumbnail_path = os.path.join(output_folder, os.path.basename(output_file).replace(".mp4", ".jpg"))
            if os.path.exists(thumbnail_path):
                os.remove(thumbnail_path)

            generate_thumbnail(movie_file, output_folder)  # Use only the movie file

        finally:
            # Cleanup temporary files
            for f in [movie_clip, parkour_clip]:
                if os.path.exists(f):
                    os.remove(f)

        print(f"Completed video {i + 1}/x.\n")

    print("All x videos are done!")

def ffmpeg_extract(input_file, start_time, output_file):
    """Extract a clip using ffmpeg and ensure audio stays in sync."""
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_time),
        '-i', input_file,
        '-t', str(CLIP_DURATION),
        '-c:v', 'libx264',  # Re-encode video to ensure sync
        '-preset', 'ultrafast',  # Speed up processing
        '-crf', '23',  # Good balance of quality and size
        '-c:a', 'aac',  # Ensure audio compatibility
        '-b:a', '192k',  # Set higher audio bitrate
        '-async', '1',  # Attempt to fix any sync issues
        output_file
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error extracting clip from {input_file}: {e}")
        sys.exit(1)


def stack_videos(top_video, bottom_video, output_folder, movie_file):
    """Stack two videos vertically, then crop dynamically based on detected faces."""
    movie_name = os.path.splitext(os.path.basename(movie_file))[0]  # Keep original movie title
    movie_name = movie_name.replace(" ", "_")  # Remove spaces for clean filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Unique timestamp
    output_file = os.path.join(output_folder, f"{movie_name}_{timestamp}.mp4")  # Stacked video name

    # Step 1: Stack Videos
    cmd_stack = [
        'ffmpeg', '-y',
        '-threads', '4',
        '-i', top_video,
        '-i', bottom_video,
        '-filter_complex', '[0:v]fps=30[v0];[1:v]fps=30[v1];[v0][v1]vstack=inputs=2[v]',
        '-map', '[v]',
        '-map', '0:a?',
        '-map', '1:a?',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-crf', '28',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-shortest',
        output_file
    ]

    try:
        subprocess.run(cmd_stack, check=True)
        print(f"Final stacked video created: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error stacking videos: {e}")
        sys.exit(1)

    # Step 2: Crop Video Based on Faces
    cropped_video = crop_video_based_on_faces(output_file, output_folder)

    if cropped_video:
        output_file = cropped_video  # Use cropped version

    # Step 3: Split into 2-minute segments
    split_output_template = os.path.join(output_folder, f"{movie_name}_part_%02d.mp4")
    
    cmd_split = [
        'ffmpeg', '-y',
        '-i', output_file,
        '-c', 'copy',
        '-map', '0',
        '-segment_time', '120',  # 2-minute segments
        '-f', 'segment',
        '-reset_timestamps', '1',
        split_output_template
    ]

    try:
        subprocess.run(cmd_split, check=True)
        print(f"Video split into 2-minute segments: {split_output_template}")
    except subprocess.CalledProcessError as e:
        print(f"Error splitting video: {e}")
        sys.exit(1)

    return output_file  # Return the cropped & segmented file

if __name__ == "__main__":
    main()