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

def pick_random_file(folder, min_length=CLIP_DURATION):
    """Pick a random video file that is at least the required length."""
    files = find_video_files(folder)
    valid_files = [f for f in files if get_duration(f) >= min_length]
    if not valid_files:
        print(f"No valid video files longer than {min_length} seconds found in {folder}")
        sys.exit(1)
    return random.choice(valid_files)

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

def extract_keyframes(trimmed_video, output_folder):
    """Extracts keyframes from the final 10-minute clip, not the original full video."""
    keyframes_folder = os.path.join(output_folder, "keyframes")
    os.makedirs(keyframes_folder, exist_ok=True)

    cmd = [
        'ffmpeg', '-y',
        '-i', trimmed_video,  # ⬅️ Now using the trimmed 10-minute clip!
        '-vf', "select='eq(pict_type\\,I)',scale=1280:720",
        '-vsync', 'vfr',
        os.path.join(keyframes_folder, "frame_%04d.jpg")
    ]

    try:
        print(f"🔹 Extracting keyframes from final 10-minute clip: {trimmed_video}...")
        subprocess.run(cmd, check=True)
        print(f"✅ Keyframes saved in: {keyframes_folder}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error extracting keyframes: {e}")
        return None

    return keyframes_folder

import cv2
import numpy as np

def calculate_sharpness(image):
    """Measure image sharpness using the Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def choose_best_thumbnail(keyframes_folder):
    """Select the best keyframe by prioritizing a clear, sharp face with open eyes."""
    keyframe_files = sorted(os.listdir(keyframes_folder))
    
    if not keyframe_files:
        print("No keyframes found.")
        return None

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")  # Added eye detection

    best_face_frame = None
    best_score = 0  # Score based on sharpness & face size

    for file in keyframe_files:
        frame_path = os.path.join(keyframes_folder, file)
        img = cv2.imread(frame_path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            face_area = w * h  # Bigger face = better
            sharpness = calculate_sharpness(img)

            # Detect eyes inside the face region
            face_roi = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=3)

            # Score: (bigger faces + more sharpness), ignore frames where no eyes are found (likely blinking)
            score = face_area * 0.3 + sharpness * 0.7
            if len(eyes) > 0 and score > best_score:
                best_score = score
                best_face_frame = frame_path

    if best_face_frame:
        print(f"✅ Selected best frame: {best_face_frame} (Sharp & Good Face)")
        return best_face_frame
    else:
        print("⚠️ No ideal face found, using brightest frame instead.")
        return None  # Fallback to brightness logic if needed

# detect_faces_in_video is deprecated for now
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

# crop_video_based_on_faces is deprecated for now
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
def get_image_dimensions(image_path):
    """Get the width and height of an image using FFmpeg."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0",
        image_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        width, height = map(int, result.stdout.strip().split(","))
        return width, height
    except Exception as e:
        print(f"Error getting image dimensions: {e}")
        return None, None

import shutil  # Make sure shutil is imported

def generate_thumbnail(movie_file, output_folder):
    """Extracts keyframes, selects the best one as a thumbnail, applies vignette + border, and cleans up."""
    keyframes_folder = extract_keyframes(movie_file, output_folder)

    if not keyframes_folder:
        print("No keyframes found, using default middle frame.")
        return extract_middle_frame(movie_file, output_folder)  # Fallback

    best_frame = choose_best_thumbnail(keyframes_folder)

    if best_frame:
        final_thumbnail = os.path.join(output_folder, os.path.basename(movie_file).replace(".mp4", ".jpg"))
        enhanced_thumbnail = final_thumbnail.replace(".jpg", "_enhanced.jpg")

        # Apply vignette & border, ensuring proper format
        cmd = [
            "ffmpeg", "-y", "-i", best_frame,
            "-vf", "vignette=PI/4, pad=width=iw+60:height=ih+60:x=30:y=30:color=black, format=yuvj420p",
            "-update", "1", "-frames:v", "1",
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
    if os.path.exists(keyframes_folder):
        shutil.rmtree(keyframes_folder, ignore_errors=True)
        print(f"Deleted keyframes folder: {keyframes_folder}")

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
    
    for i in range(1):  # Run X times

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

import subprocess

def ffmpeg_extract(input_file, start_time, output_file, clip_duration=600):
    """Extract a clip using ffmpeg, avoiding unnecessary re-encoding when possible."""
    
    # Check the codec of the input file
    cmd_check = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name", "-of", "csv=p=0", input_file
    ]
    
    codec_name = subprocess.run(cmd_check, capture_output=True, text=True).stdout.strip()

    # ✅ If the video is already H.264, avoid re-encoding
    if codec_name == "h264":
        print(f"✅ Copying clip without re-encoding: {input_file}...")
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", input_file,
            "-t", str(clip_duration),
            "-c:v", "copy", "-c:a", "copy",
            output_file
        ]
    else:
        print(f"🔹 Extracting & re-encoding clip: {input_file}...")
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", input_file,
            "-t", str(clip_duration),
            "-c:v", "libx264", "-crf", "19", "-preset", "fast",
            "-c:a", "aac", "-b:a", "128k",
            "-threads", "4",
            output_file
        ]

    subprocess.run(cmd, check=True)

import glob

def stack_videos(top_video, bottom_video, output_folder, movie_file):
    """Stack two videos vertically, transcribe, generate a title, and split into numbered parts."""
    movie_name = os.path.splitext(os.path.basename(movie_file))[0].replace(" ", "_")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_folder, f"{movie_name}_{timestamp}.mp4")

    # Step 1: Resize Videos
    def get_video_width(video_path):
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width", "-of", "csv=p=0", video_path]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return int(result.stdout.strip())
        except Exception as e:
            print(f"Error getting width of {video_path}: {e}")
            return None

    top_width, bottom_width = get_video_width(top_video), get_video_width(bottom_video)
    if not top_width or not bottom_width:
        print("Error determining video widths.")
        sys.exit(1)

    target_width = min(top_width, bottom_width)
    resized_top, resized_bottom = os.path.join(output_folder, "resized_top.mp4"), os.path.join(output_folder, "resized_bottom.mp4")

    import subprocess

    def resize_video(input_file, output_file, width):
        """Resize video only if necessary, avoiding unnecessary re-encoding."""
        
        # Check if video is already the correct width
        cmd_check = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,codec_name", "-of", "csv=p=0", input_file
        ]
        
        video_info = subprocess.run(cmd_check, capture_output=True, text=True).stdout.strip()
        width_check, codec_name = video_info.split("\n")[0].split(",")  # Extract width & codec

        if width_check == str(width) and codec_name == "h264":
            # ✅ If already correct width and H.264, just copy (NO re-encoding!)
            print(f"✅ Skipping resize, copying {input_file}...")
            cmd = ["ffmpeg", "-y", "-i", input_file, "-c:v", "copy", "-c:a", "copy", output_file]
        else:
            # 🔹 If resizing is needed, use ultrafast CPU encoding OR GPU encoding if available
            print(f"🔹 Resizing {input_file} to width {width}...")

            # **Fastest Option: Use GPU (if available)**
            gpu_available = False  # Change to True if you have an NVIDIA GPU and want to use NVENC
            if gpu_available:
                cmd = [
                    "ffmpeg", "-y", "-hwaccel", "cuda", "-i", input_file,
                    "-vf", f"scale={width}:-2",
                    "-c:v", "h264_nvenc", "-preset", "fast", "-b:v", "5M",
                    "-c:a", "aac", "-b:a", "128k",
                    output_file
                ]
            else:
                # **Fallback: Fastest CPU-based encoding**
                cmd = [
                    "ffmpeg", "-y", "-i", input_file,
                    "-vf", f"scale={width}:-2",
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "19",
                    "-c:a", "aac", "-b:a", "128k",
                    "-threads", "8",  # Use more threads if available
                    output_file
                ]

        subprocess.run(cmd, check=True)

    resize_video(top_video, resized_top, target_width)
    resize_video(bottom_video, resized_bottom, target_width)

    # Step 2: Stack Videos
    cmd_stack = [
        'ffmpeg', '-y',
        '-threads', '4',
        '-i', resized_top,
        '-i', resized_bottom,
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
        print(f"✅ Final stacked video created: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error stacking videos: {e}")
        sys.exit(1)
    finally:
        for f in [resized_top, resized_bottom]:
            if os.path.exists(f):
                os.remove(f)
                print(f"🗑 Deleted temporary file: {f}")

    # Step 3: 🎙 Transcribe video & generate title BEFORE splitting
    transcript = transcribe_video_whisperx(output_file)
    generated_title = generate_final_title(movie_file, transcript)
    safe_title = re.sub(r'[^\w\s-]', '', generated_title).replace(" ", "_")  # Remove special chars

    # Step 4: Split Video into 2-Minute Segments
    split_output_template = os.path.join(output_folder, f"{safe_title}_part_%02d.mp4")

    cmd_split = ["ffmpeg", "-y", "-i", output_file, "-c", "copy", "-map", "0", "-segment_time", "120", "-f", "segment", "-reset_timestamps", "1", split_output_template]

    try:
        print(f"🔹 Running split command: {' '.join(cmd_split)}")
        subprocess.run(cmd_split, check=True)
        print(f"✅ Video split into 2-minute segments: {split_output_template}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error splitting video: {e}")
        sys.exit(1)

    split_files = sorted(glob.glob(os.path.join(output_folder, f"{safe_title}_part_*.mp4")))

    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"🗑 Deleted full video: {output_file}")

    # Step 5: Rename Segments for Consistency
    for index, segment_file in enumerate(sorted(split_files), start=1):
        new_name = os.path.join(output_folder, f"{safe_title}_Part_{index}.mp4")
        os.rename(segment_file, new_name)
        print(f"🔄 Renamed {segment_file} -> {new_name}")

    return output_file  # Return the stacked file

import re
from collections import Counter

def generate_title_from_filename(movie_file):
    """Generate a title from the original filename."""
    base_name = os.path.basename(movie_file).replace(".mp4", "")
    cleaned_title = re.sub(r'[_\-0-9]', ' ', base_name)  # Remove special chars
    return ' '.join([word.capitalize() for word in cleaned_title.split()])  # Capitalize words

def generate_title_from_transcript(transcript_text, weight=0.3):
    """Extract top keywords from transcript but limit their influence."""
    words = re.findall(r'\b\w+\b', transcript_text.lower())  # Extract words
    ignore_words = {"the", "and", "to", "is", "of", "a", "in", "that", "it", "on", "with", "this", "for"}
    filtered_words = [word for word in words if word not in ignore_words]
    common_words = Counter(filtered_words).most_common(5)  # Get top 5 keywords

    # Keep only a percentage of transcript-based words (e.g., 30% weight)
    num_to_include = max(1, int(len(common_words) * weight))  # Ensure at least 1 word
    transcript_keywords = ' '.join([word.capitalize() for word, _ in common_words[:num_to_include]])

    return transcript_keywords

def generate_final_title(movie_file, transcript_text=None):
    """Prioritize the movie filename and add transcript keywords only if useful."""
    filename_title = generate_title_from_filename(movie_file)

    if transcript_text:
        transcript_title = generate_title_from_transcript(transcript_text)

        # Only add transcript keywords if they add new meaning
        if transcript_title and transcript_title.lower() not in filename_title.lower():
            return f"{filename_title} – {transcript_title}"  # Keep the filename as the main part

    return filename_title  # Default to just the filename if transcript is weak

import whisperx
import torch

def transcribe_video_whisperx(video_path):
    """Transcribe audio from a video using WhisperX."""
    print(f"🎙 Transcribing: {video_path}...")

    # Load WhisperX model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Force float32 precision to avoid the error
    compute_type = "float32" if device == "cpu" else "float16"
    model = whisperx.load_model("base", device, compute_type=compute_type)

    # Transcribe audio
    audio = whisperx.load_audio(video_path)
    result = model.transcribe(audio)

    # Extract transcript text
    transcript_text = " ".join([seg["text"] for seg in result["segments"]])

    print(f"✅ Transcription complete: {len(transcript_text)} characters")
    return transcript_text

if __name__ == "__main__":
    main()