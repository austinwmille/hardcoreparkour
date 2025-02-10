import os
import json
import time
from PIL import Image, ImageDraw, ImageFont
import subprocess
from datetime import timedelta
import torch
import whisperx
from dotenv import load_dotenv
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from openai import OpenAI

# ---------------------------------------------------------------- #
# our folder path
VIDEO_FOLDER_PATH = "../random clips"

load_dotenv('mushroom.env')

# Load OpenAI API Key from environment variable
OPENAI_API_KEY = os.getenv("gptkey")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API Key is not set. Please set it as an environment variable.")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Path to your OAuth 2.0 Client ID JSON file
CLIENT_SECRET_FILE = 'mushroom.json'
SCOPES = ['https://www.googleapis.com/auth/youtube.upload']

# Authenticate with YouTube API
def get_authenticated_service():
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
    credentials = flow.run_local_server(port=0)
    return build('youtube', 'v3', credentials=credentials)

# -------------------------------------------------------------------- #

# Transcribe video using Whisper
def transcribe_video(file_path):
    try:
        import whisper
        model = whisper.load_model("medium")  # Use 'small' or 'medium' for better accuracy
        result = model.transcribe(file_path)
        return result["text"]
    except Exception as e:
        print(f"Error transcribing video: {e}")
        return ""

# Call OpenAI API to generate metadata for a provocative right-wing news clip.
def generate_news_metadata(file_path):
    transcript = transcribe_video(file_path) or "No transcript available."
    original_file_name = os.path.splitext(os.path.basename(file_path))[0]

    prompt = (
        f"Generate YouTube metadata as JSON (title, description, tags, category) for a provocative right-wing news clip. RULES:\n"
        f"1. TITLE:\n"
        f"   - Reword the original title (derived from the filename) in a straightforward and neutral manner. Only include hook words if they clearly add value.\n"
        f"2. DESCRIPTION:\n"
        f"   - Provide a simple, clear description that is not too dramatic.\n"
        f"   - Do not include generic calls to action except an optional brief note like 'Subscribe for more digestible clips every week!'.\n"
        f"3. TAGS:\n"
        f"   - Include relevant tags that reflect interviews, debates, and news content without overusing trendy terms.\n"
        f"4. CATEGORY:\n"
        f"   - Set the category to 'News'.\n"
        f"---\n"
        f"IMPORTANT: Return only a valid JSON object with no markdown formatting or additional text.\n"
        f"Filename: {original_file_name}\n"
        f"Transcript: {transcript}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Replace with your intended model (e.g., "gpt-4" or "gpt-3.5-turbo")
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant specialized in YouTube metadata optimization for news content."
                },
                {"role": "user", "content": prompt},
            ]
        )
        metadata_content = response.choices[0].message.content.strip()

        # Remove markdown formatting (triple backticks) if present:
        import re
        metadata_content = re.sub(r"^```(?:json)?", "", metadata_content)
        metadata_content = re.sub(r"```$", "", metadata_content)
        metadata_content = metadata_content.strip()

        # Parse the cleaned string into JSON
        metadata = json.loads(metadata_content)

        # Force the category to be 'News'
        metadata['category'] = 'News'
        metadata['categoryId'] = '25'
        return metadata

    except json.JSONDecodeError:
        print("AI response not in JSON format. Using default metadata.")
        with open('description.txt', 'r', encoding='utf-8') as file:
            description_text = file.read().strip()
        return {
            "title": "Don't like; testing 1",
            "description": description_text,
            "tags": [
                "birds", "parakeets", "budgies", "parakeet", 
                "budgie", "parrot", "parrake", "bird", "fly", "flying", "sky"
            ],
            "categoryId": "25"
        }
    except Exception as e:
        print(f"Error generating metadata: {e}")
        with open('description.txt', 'r', encoding='utf-8') as file:
            description_text = file.read().strip()
        return {
            "title": "Don't like; testing 1",
            "description": description_text,
            "tags": [
                "birds", "parakeets", "budgies", "parakeet", 
                "budgie", "parrot", "parrake", "bird", "fly", "flying", "sky"
            ],
            "categoryId": "25"
        }

def upload_video(youtube, file_path, metadata):
    # Use metadata description if available; otherwise, fall back to reading description.txt.
    description_text = metadata.get('description')
    if not description_text:
        try:
            with open('description.txt', 'r', encoding='utf-8') as file:
                description_text = file.read().strip()
        except Exception as e:
            print(f"Error reading description.txt: {e}")
            description_text = "No description provided."

    request_body = {
        'snippet': {
            'title': metadata.get('title', "pls don't watch, testing"),
            'description': metadata.get('description', description_text),
            'tags': metadata.get('tags', [
                "birds", "parakeets", "budgies", "parakeet", 
                "budgie", "parrot", "parrake", "bird", "fly", "flying", "sky"
            ]),
            'categoryId': "25",  # Force category to 'News'
        },
        'status': {
            'privacyStatus': 'public',  # Options: 'public', 'unlisted', 'private'
            'selfDeclaredMadeForKids': False  # Default to "Not Made for Kids"
        }
    }

    media_file = MediaFileUpload(file_path, chunksize=-1, resumable=True)
    request = youtube.videos().insert(
        part="snippet,status",
        body=request_body,
        media_body=media_file
    )

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"Uploading {file_path}: {int(status.progress() * 100)}% complete.")
    print(f"Upload completed: {metadata.get('title', 'Untitled Video')}")
    return response

# ASS Karaoke Subtitle Generation & Burning Functions
def format_ass_timestamp(seconds):
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    centiseconds = int((seconds - total_seconds) * 100)
    return f"{hours}:{minutes:02}:{secs:02}.{centiseconds:02}"

def generate_ass_karaoke_subtitles(aligned_segments, ass_path, max_words=3, pause_threshold=0.3):
    """
    Generate an ASS subtitle file using karaoke timing.
    Groups word-level alignments into blocks based on a maximum number of words or a pause threshold.
    For each block, each word is prefixed with karaoke tags plus an override for a pop effect.
    """
    header = (
        "[Script Info]\n"
        "Title: Karaoke Subtitles\n"
        "ScriptType: v4.00+\n"
        "Collisions: Normal\n"
        "PlayResX: 1280\n"
        "PlayResY: 2160\n"  # Adjust if your resolution differs.
        "Timer: 100.0000\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,"
        "Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
        "Alignment,MarginL,MarginR,MarginV,Encoding\n"
        # Style: Arial, size 36, primary colour = #6A0DAD (ASS BGR: &H00AD0D6A),
        # Alignment=2 (bottom-center) with MarginV=300 (which you can adjust to push them higher).
        "Style: Default,Arial,36,&H00AD0D6A,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,2,2,10,10,300,1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    
    words_list = []
    for segment in aligned_segments:
        if "words" in segment and segment["words"]:
            for w in segment["words"]:
                start = w.get("start", w.get("start_time"))
                end = w.get("end", w.get("end_time"))
                if start is None:
                    continue
                if end is None and "duration" in w and start is not None:
                    end = start + w["duration"]
                if end is None:
                    continue
                words_list.append({
                    "start": start,
                    "end": end,
                    "word": w.get("word", "")
                })
    words_list.sort(key=lambda x: x["start"])
    
    blocks = []
    current_block = []
    for w in words_list:
        if not current_block:
            current_block.append(w)
        else:
            last_word = current_block[-1]
            gap = w["start"] - last_word["end"]
            if gap > pause_threshold or len(current_block) >= max_words:
                blocks.append(current_block)
                current_block = [w]
            else:
                current_block.append(w)
    if current_block:
        blocks.append(current_block)
    
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header)
        for block in blocks:
            if not block:
                continue
            block_start = block[0]["start"]
            block_end = block[-1]["end"]
            karaoke_text = ""
            n = len(block)
            for i, word in enumerate(block):
                word_text = word["word"]
                if i < n - 1:
                    duration_cs = int(round((block[i+1]["start"] - word["start"]) * 100))
                else:
                    duration_cs = int(round((word["end"] - word["start"]) * 100))
                karaoke_text += f"{{\\t(0,250,\\fscx170,\\fscy170)}}{{\\k{duration_cs}}}{word_text} "
            karaoke_text = karaoke_text.strip()
            line = (
                f"Dialogue: 0,{format_ass_timestamp(block_start)},{format_ass_timestamp(block_end)},Default,,0,0,0,,{karaoke_text}\n"
            )
            f.write(line)
    print(f"ASS karaoke subtitle file generated: {ass_path}")

def burn_subtitles(input_video, ass_file, output_video):
    command = [
        "ffmpeg",
        "-y",  # Overwrite output if it exists.
        "-i", input_video,
        "-vf", f"subtitles={ass_file}",
        output_video
    ]
    print("Running FFmpeg command:")
    print(" ".join(command))
    subprocess.run(command, check=True)
    print(f"Subtitled video created: {output_video}")

# Create a custom thumbnail by overlaying a black box with text on the base thumbnail.
def create_thumbnail(base_thumbnail_path, part_text, output_path):
    try:
        img = Image.open(base_thumbnail_path).convert("RGBA")
    except Exception as e:
        print(f"Error opening base thumbnail: {e}")
        return None

    width, height = img.size
    box_width, box_height = 200, 50
    box_x = (width - box_width) // 2
    box_y = height // 2 + 20  # Just below the middle

    draw = ImageDraw.Draw(img)
    black_color = (0, 0, 0, 200)
    draw.rectangle([box_x, box_y, box_x + box_width, box_y + box_height], fill=black_color)

    text = part_text
    font_size = 28
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    bbox = font.getbbox(text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = box_x + (box_width - text_width) // 2
    text_y = box_y + (box_height - text_height) // 2

    draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255, 255))
    try:
        img.convert("RGB").save(output_path, "JPEG")
    except Exception as e:
        print(f"Error saving thumbnail: {e}")
        return None
    return output_path

# Set the generated thumbnail for a video via the YouTube API.
def set_thumbnail(youtube, video_id, thumbnail_path):
    try:
        response = youtube.thumbnails().set(
            videoId=video_id,
            media_body=MediaFileUpload(thumbnail_path)
        ).execute()
        print(f"Thumbnail set for video {video_id}")
    except Exception as e:
        print(f"Error setting thumbnail for video {video_id}: {e}")

# Process all video files in the specified folder (only files in the root, not in subfolders).
def process_folder(youtube, folder_path):
    base_thumbnail_file = None
    for f in os.listdir(folder_path):
        if f.lower().endswith(".jpg"):
            base_thumbnail_file = os.path.join(folder_path, f)
            break
    if not base_thumbnail_file:
        print("No base thumbnail file (.jpg) found in folder.")

    video_files = sorted(
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))
    )

    if not video_files:
        print("No video files found in folder:", folder_path)
        return

    part_number = 1
    for file_path in video_files:
        print(f"Processing video: {file_path}")
        metadata = generate_news_metadata(file_path)
        print(f"Generated Metadata: {metadata}")

        # First, generate the karaoke subtitles for the video.
        # We'll use WhisperX for forced alignment and generate an ASS file.
        # Load the WhisperX model and alignment model.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device} for subtitle generation")
        pipeline = whisperx.load_model("medium", device, compute_type="float32")
        print(f"Transcribing video for subtitles: {file_path}")
        result = pipeline.transcribe(file_path)
        language = result.get("language", "en")
        print(f"Detected language for subtitles: {language}")
        alignment_model, metadata_align = whisperx.load_align_model(language, device)
        print("Running forced alignment for subtitles...")
        result_aligned = whisperx.align(result["segments"], alignment_model, metadata_align, file_path, device)
        print("Forced alignment complete for subtitles.")
        ass_file = os.path.join(folder_path, f"temp_karaoke_{part_number}.ass")
        print("Generating ASS karaoke subtitle file...")
        generate_ass_karaoke_subtitles(result_aligned["segments"], ass_file, max_words=3, pause_threshold=0.3)
        subbed_video = os.path.join(folder_path, f"temp_subbed_{part_number}.mp4")
        print("Burning subtitles into the video...")
        burn_subtitles(file_path, ass_file, subbed_video)
        os.remove(ass_file)  # Remove temporary ASS file after burning subtitles.

        # Now, upload the subbed video.
        response = upload_video(youtube, subbed_video, metadata)
        video_id = response.get("id")
        print(f"Uploaded {file_path} (subbed version) with video ID: {video_id}")

        # Generate and set a custom thumbnail if possible.
        if base_thumbnail_file and video_id:
            output_thumbnail = os.path.join(folder_path, f"temp_thumbnail_{part_number}.jpg")
            part_text = f"part {part_number}"
            created_thumbnail = create_thumbnail(base_thumbnail_file, part_text, output_thumbnail)
            if created_thumbnail:
                set_thumbnail(youtube, video_id, created_thumbnail)
            if os.path.exists(output_thumbnail):
                os.remove(output_thumbnail)
        else:
            print("Skipping thumbnail generation due to missing base thumbnail or video ID.")

        print("Waiting 10 minutes before the next upload.")
        time.sleep(600)  # Wait 10 minutes (600 seconds)
        # Optionally, remove the temporary subbed video after upload if desired.
        if os.path.exists(subbed_video):
            os.remove(subbed_video)
        part_number += 1

# Main function
def main():
    youtube = get_authenticated_service()
    if os.path.exists(VIDEO_FOLDER_PATH) and os.path.isdir(VIDEO_FOLDER_PATH):
        process_folder(youtube, VIDEO_FOLDER_PATH)
    else:
        print(f"Folder not found: {VIDEO_FOLDER_PATH}")

if __name__ == '__main__':
    main()
