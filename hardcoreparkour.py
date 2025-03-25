import os
import random
import subprocess
import sys
import tempfile
import datetime
import cv2
import numpy as np
import shutil
import glob
import re
from collections import Counter
import torch
import whisperx

# Configuration
MOVIE_FOLDER = "./gou/"
PARKOUR_FOLDER = "./minecraft parkour vids/"
OUTPUT_FOLDER = "./"  # This will be used inside the stack_videos function
CLIP_DURATION = 600 #length in seconds

def find_video_files(folder):
    """Return a list of video files in the given folder (non-recursive)."""
    video_extensions = ('.mp4', '.mov', '.mkv')
    video_files = []
    for file in os.listdir(folder):
        full_path = os.path.join(folder, file)
        if os.path.isfile(full_path) and file.lower().endswith(video_extensions):
            video_files.append(full_path)
    return video_files
    
def pick_random_file(folder, min_length=CLIP_DURATION):
    """Pick a random video file from the given folder."""
    files = find_video_files(folder)
    if not files:
        print(f"No video files found in {folder}")
        sys.exit(1)
    return random.choice(files)

def get_duration(file_path):
    """Get duration of video file in seconds using ffprobe."""
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
    """Pick a random start time for the clip."""
    max_start = total_duration - CLIP_DURATION
    if max_start <= 0:
        return 0
    return random.randint(0, max_start)

def ffmpeg_extract(input_file, start_time, output_file):
    """Extract a clip using ffmpeg, avoiding unnecessary re-encoding when possible."""
    cmd_check = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name", "-of", "csv=p=0", input_file
    ]
    codec_name = subprocess.run(cmd_check, capture_output=True, text=True).stdout.strip()

    if codec_name == "h264":
        print(f"✅ Copying clip without re-encoding: {input_file}...")
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", input_file,
            "-t", str(CLIP_DURATION),
            "-c:v", "copy", "-c:a", "copy",
            output_file
        ]
    else:
        print(f"🔹 Extracting & re-encoding clip: {input_file}...")
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", input_file,
            "-t", str(CLIP_DURATION),
            "-c:v", "libx264", "-crf", "19", "-preset", "medium",
            "-c:a", "aac", "-b:a", "160k",
            "-threads", "4",
            output_file
        ]
    subprocess.run(cmd, check=True)

def extract_keyframes(trimmed_video, output_folder, movie_file):
    """Extracts keyframes from the final clip and deletes the original movie file."""
    keyframes_folder = os.path.join(output_folder, "keyframes")
    os.makedirs(keyframes_folder, exist_ok=True)
    cmd = [
        'ffmpeg', '-y',
        '-i', trimmed_video,
        '-vf', "select='eq(pict_type\\,I)',scale=1920:1080:flags=lanczos",
        '-vsync', 'vfr',
        os.path.join(keyframes_folder, "frame_%04d.jpg")
    ]
    try:
        print(f"Extracting keyframes from clip: {trimmed_video}...")
        subprocess.run(cmd, check=True)
        print(f"✅ Keyframes saved in: {keyframes_folder}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error extracting keyframes: {e}")
        return None
    if os.path.exists(movie_file):
        try:
            os.remove(movie_file)
            print(f"🗑 Deleted original movie file: {movie_file}")
        except Exception as e:
            print(f"Error deleting original movie file: {e}")
    return keyframes_folder

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
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    best_face_frame = None
    best_score = 0
    for file in keyframe_files:
        frame_path = os.path.join(keyframes_folder, file)
        img = cv2.imread(frame_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        for (x, y, w, h) in faces:
            face_area = w * h
            sharpness = calculate_sharpness(img)
            face_roi = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=3)
            score = face_area * 0.3 + sharpness * 0.7
            if len(eyes) > 0 and score > best_score:
                best_score = score
                best_face_frame = frame_path
    if best_face_frame:
        print(f"✅ Selected best frame: {best_face_frame} (Sharp & Good Face)")
        return best_face_frame
    else:
        print("⚠️ No ideal face found, using brightest frame instead.")
        return None

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

def generate_thumbnail(movie_file, output_folder):
    """Extracts keyframes, selects the best one as a thumbnail, applies vignette + border, and cleans up."""
    trimmed_video = movie_file
    keyframes_folder = extract_keyframes(trimmed_video, output_folder, movie_file)
    if not keyframes_folder:
        print("No keyframes found, using default middle frame.")
        return extract_middle_frame(movie_file, output_folder)
    best_frame = choose_best_thumbnail(keyframes_folder)
    if best_frame:
        basename = os.path.splitext(os.path.basename(movie_file))[0]
        final_thumbnail = os.path.join(output_folder, f"{basename}.jpg")
        enhanced_thumbnail = final_thumbnail.replace(".jpg", "_enhanced.jpg")
        cmd = [
            "ffmpeg", "-y", "-i", best_frame,
            "-vf", "vignette=PI/4,eq=contrast=1.3:brightness=0.1:saturation=1.2,unsharp=3:3:1.0:3:3:0.0,pad=width=iw+60:height=ih+60:x=30:y=30:color=black,format=yuvj420p",
            "-update", "1", "-frames:v", "1",
            enhanced_thumbnail
        ]
        try:
            subprocess.run(cmd, check=True)
            print(f"Thumbnail saved: {enhanced_thumbnail}")
            final_thumbnail = enhanced_thumbnail
        except subprocess.CalledProcessError as e:
            print(f"Error applying vignette/border: {e}")
    else:
        print("No suitable keyframes found, using default middle frame.")
        final_thumbnail = extract_middle_frame(movie_file, output_folder)
    if os.path.exists(keyframes_folder):
        shutil.rmtree(keyframes_folder, ignore_errors=True)
        print(f"Deleted keyframes folder: {keyframes_folder}")
    return final_thumbnail

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

def stack_videos(top_video, bottom_video, output_folder, movie_file):
    """Stack two videos vertically, transcribe, generate a title, and split into numbered parts."""
    movie_name = os.path.splitext(os.path.basename(movie_file))[0].replace(" ", "_")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_folder, f"{movie_name}_{timestamp}.mp4")

    def get_video_width(video_file):
        """Returns the width of the video using ffprobe."""
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,codec_name",
                "-of", "csv=p=0",
                video_file
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            parts = [p.strip() for p in result.stdout.strip().split(",") if p.strip()]
            for part in parts:
                try:
                    return int(part)
                except ValueError:
                    continue
            raise ValueError("No valid width found in ffprobe output.")
        except Exception as e:
            print(f"⚠️ Error getting width of {video_file}: {e}")
            return None

    top_width, bottom_width = get_video_width(top_video), get_video_width(bottom_video)
    if not top_width or not bottom_width:
        print("Error determining video widths.")
        sys.exit(1)

    target_width = min(top_width, bottom_width)
    resized_top = os.path.join(output_folder, "resized_top.mp4")
    resized_bottom = os.path.join(output_folder, "resized_bottom.mp4")

    def resize_video(input_file, output_file, width):
        # Resize video only if necessary, avoiding unnecessary re-encoding.
        cmd_check = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,codec_name", "-of", "csv=p=0",
            input_file
        ]
        video_info = subprocess.run(cmd_check, capture_output=True, text=True).stdout.strip()
        line = video_info.split("\n")[0].rstrip(',').strip()
        parts = [part.strip() for part in line.split(",") if part.strip()]
        if len(parts) < 2:
            raise ValueError("Unexpected ffprobe output format.")
        width_check, codec_name = parts[0], parts[1]
        if width_check == str(width) and codec_name == "h264":
            print(f"✅ Skipping resize, copying {input_file}...")
            cmd = ["ffmpeg", "-y", "-i", input_file, "-c:v", "copy", "-c:a", "copy", output_file]
        else:
            print(f"🔹 Resizing {input_file} to width {width}...")
            gpu_available = False  # Change to True if GPU encoding is available
            if gpu_available:
                cmd = [
                    "ffmpeg", "-y", "-hwaccel", "cuda", "-i", input_file,
                    "-vf", f"scale={width}:-2",
                    "-c:v", "h264_nvenc", "-preset", "fast", "-b:v", "5M",
                    "-c:a", "aac", "-b:a", "160k",
                    output_file
                ]
            else:
                cmd = [
                    "ffmpeg", "-y", "-i", input_file,
                    "-vf", f"scale={width}:-2",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "19",
                    "-c:a", "aac", "-b:a", "160k",
                    "-threads", "4",
                    output_file
                ]
        subprocess.run(cmd, check=True)

    # Now call resize_video for both videos (outside the function definition)
    resize_video(top_video, resized_top, target_width)
    resize_video(bottom_video, resized_bottom, target_width)

    # Add this helper function inside stack_videos (or at the module level)
    def get_video_height(video_file):
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=height",
                "-of", "csv=p=0",
                video_file
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return int(result.stdout.strip())
        except Exception as e:
            print(f"Error getting height of {video_file}: {e}")
            return None

    # Set final output resolution to 9:16 (for example, 1080x1920)
    final_width = 1080
    final_height = 1920

    # Instead of computing crop heights based on target ratios,
    # we now force each clip to be cropped to 1080x960.
    top_crop_height = 960
    bottom_crop_height = 960

    # Build the filter_complex with the new values:
    filter_complex = (
        f"[0:v]fps=30,crop={final_width}:{top_crop_height}:0:((in_h-{top_crop_height})/2)[v0]; "
        f"[1:v]fps=30,crop={final_width}:{bottom_crop_height}:0:((in_h-{bottom_crop_height})/2)[v1]; "
        f"[v0][v1]vstack=inputs=2,scale={final_width}:{final_height}[v]"
    )

    cmd_stack = [
        'ffmpeg', '-y',
        '-threads', '4',
        '-i', resized_top,
        '-i', resized_bottom,
        '-filter_complex', filter_complex,
        '-map', '[v]',
        '-map', '0:a?',
        '-map', '1:a?',
        '-c:v', 'libx264',
        '-preset', 'slow',
        '-crf', '16',  # Lower CRF for higher quality
        '-c:a', 'aac',
        '-b:a', '160k',
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

    transcript = transcribe_video_whisperx(output_file)
    generated_title = generate_final_title(movie_file, transcript)
    safe_title = re.sub(r'[^\w\s-]', '', generated_title).replace(" ", "_")
    split_output_template = os.path.join(output_folder, f"{safe_title}_part_%02d.mp4")
    cmd_split = ["ffmpeg", "-y", "-i", output_file, "-c", "copy", "-map", "0",
                 "-segment_time", "120", "-f", "segment", "-reset_timestamps", "1", split_output_template]
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
    for index, segment_file in enumerate(sorted(split_files), start=1):
        new_name = os.path.join(output_folder, f"{safe_title}_Part_{index}.mp4")
        os.rename(segment_file, new_name)
        print(f"Renamed {segment_file} -> {new_name}")
    return output_file

def generate_title_from_filename(movie_file):
    """Generate a title from the original filename."""
    base_name = os.path.basename(movie_file).replace(".mp4", "")
    cleaned_title = re.sub(r'[_\-0-9]', ' ', base_name)
    return ' '.join([word.capitalize() for word in cleaned_title.split()])

def generate_title_from_transcript(transcript_text, weight=0.3):
    """Extract top keywords from transcript but limit their influence."""
    words = re.findall(r'\b\w+\b', transcript_text.lower())
    ignore_words = {"the", "and", "to", "is", "of", "a", "in", "that", "it", "on", "with", "this", "for"}
    filtered_words = [word for word in words if word not in ignore_words]
    common_words = Counter(filtered_words).most_common(5)
    num_to_include = max(1, int(len(common_words) * weight))
    transcript_keywords = ' '.join([word.capitalize() for word, _ in common_words[:num_to_include]])
    return transcript_keywords

def generate_final_title(movie_file, transcript_text=None):
    """Prioritize the movie filename and add transcript keywords if useful."""
    filename_title = generate_title_from_filename(movie_file)
    if transcript_text:
        transcript_title = generate_title_from_transcript(transcript_text)
        if transcript_title and transcript_title.lower() not in filename_title.lower():
            return f"{filename_title} – {transcript_title}"
    return filename_title

def transcribe_video_whisperx(video_path):
    """Transcribe audio from a video using WhisperX."""
    print(f"🎙 Transcribing: {video_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float32" if device == "cpu" else "float16"
    model = whisperx.load_model("base", device, compute_type=compute_type)
    audio = whisperx.load_audio(video_path)
    result = model.transcribe(audio)
    transcript_text = " ".join([seg["text"] for seg in result["segments"]])
    print(f"✅ Transcription complete: {len(transcript_text)} characters")
    return transcript_text

def main():
    # Set your output folder where the final video and related files will be saved
    output_folder = "C:/Users/austi/Desktop/hardcoreparkour/random clips"
    
    # Create temporary file paths for the movie and parkour clips
    temp_dir = tempfile.gettempdir()
    movie_clip = os.path.join(temp_dir, f"movie_clip_{os.getpid()}.mp4")
    parkour_clip = os.path.join(temp_dir, f"parkour_clip_{os.getpid()}.mp4")

    try:
        # 1. Pick random files from each folder
        movie_file = pick_random_file(MOVIE_FOLDER)
        parkour_file = pick_random_file(PARKOUR_FOLDER)

        # 2. Get durations of the chosen files
        movie_duration = get_duration(movie_file)
        parkour_duration = get_duration(parkour_file)

        # 3. Pick random start times for each clip
        movie_start = pick_random_start(movie_duration) #change to 0 to use whole video
        parkour_start = pick_random_start(parkour_duration)

        # 4. Extract clips from both files
        ffmpeg_extract(movie_file, movie_start, movie_clip)
        ffmpeg_extract(parkour_file, parkour_start, parkour_clip)

        # 5. Stack the two clips into one video
        output_file = stack_videos(movie_clip, parkour_clip, output_folder, movie_file)

        # 6. Generate a thumbnail for the movie file
        thumbnail_path = os.path.join(output_folder, os.path.basename(output_file).replace(".mp4", ".jpg"))
        if os.path.exists(thumbnail_path):
            os.remove(thumbnail_path)
        generate_thumbnail(movie_file, output_folder)

    finally:
        # Cleanup temporary files
        for f in [movie_clip, parkour_clip]:
            if os.path.exists(f):
                os.remove(f)

    print("✅ Processing complete for this video.")

if __name__ == "__main__":
    # If a command-line argument "stack" is provided, run only the stacking function.
    if len(sys.argv) > 1 and sys.argv[1] == "stack":
        # Provide your own paths for testing.
        top_video = "resized_top.mp4"
        bottom_video = "resized_bottom.mp4"
        movie_file = "gou/20200926-Tony Hawk's 1+2 Remaster Developers React to Speedrun.mp4"
        output_folder = "C:/Users/austi/Desktop/hardcoreparkour/random clips"
        stack_videos(top_video, bottom_video, output_folder, movie_file)
    else:
        main()