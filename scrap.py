import os
import subprocess
from datetime import timedelta
import torch
import whisperx

def format_timestamp(seconds):
    """
    Convert a float number of seconds into an SRT timestamp (HH:MM:SS,ms).
    """
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    milliseconds = int((seconds - total_seconds) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

def generate_srt_youtube_style(aligned_segments, srt_path, threshold=5, overlap=2, max_gap=0.5):
    """
    Generate an SRT file with "build-up" subtitles similar to YouTube's automatic captions.
    
    This function flattens the word-level alignments from all segments, then accumulates words until a threshold
    (default 5 words) is reached or a long gap (default > 0.5 sec) is encountered. It overlaps the last few words
    (default 2) with the next block to create a gradual build-up effect.
    """
    # Flatten all words from the aligned segments.
    words_list = []
    for segment in aligned_segments:
        if "words" in segment and segment["words"]:
            for w in segment["words"]:
                start = w.get("start", w.get("start_time"))
                end = w.get("end")
                if end is None:
                    duration = w.get("duration")
                    if start is not None and duration is not None:
                        end = start + duration
                if start is None or end is None:
                    continue
                words_list.append({"start": start, "end": end, "word": w.get("word", "")})
    words_list.sort(key=lambda x: x["start"])
    
    srt_blocks = []
    current_block = []
    for w in words_list:
        if not current_block:
            current_block.append(w)
        else:
            last_word = current_block[-1]
            if w["start"] - last_word["end"] > max_gap:
                srt_blocks.append(current_block)
                current_block = [w]
            else:
                current_block.append(w)
        if len(current_block) >= threshold:
            srt_blocks.append(current_block)
            current_block = current_block[-overlap:]
    if current_block:
        srt_blocks.append(current_block)
    
    block_number = 1
    with open(srt_path, "w", encoding="utf-8") as f:
        for block in srt_blocks:
            start = block[0]["start"]
            end = block[-1]["end"]
            text_line = " ".join(w["word"] for w in block).strip()
            f.write(f"{block_number}\n")
            f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            f.write(f"{text_line}\n\n")
            block_number += 1
    print(f"SRT file generated: {srt_path}")

def burn_subtitles(input_video, srt_file, output_video):
    """
    Burn subtitles into the video using FFmpeg.
    
    The subtitles are styled with:
      - Alignment=2: Bottom-center alignment.
      - MarginV=30: 30 pixels margin from the bottom (adjust to position as desired).
      - PrimaryColour=&H0082004B&: a weeping shade of indigo (Alpha=00, Blue=82, Green=00, Red=4B).
      - Outline=1 and OutlineColour=&H00000000&: a 1-pixel black outline.
      - Shadow=1: a subtle shadow.
    """
    style = ("Alignment=2,MarginV=30,PrimaryColour=&H0082004B&,"
             "Outline=1,OutlineColour=&H00000000&,Shadow=1")
    command = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists.
        "-i", input_video,
        "-vf", f"subtitles={srt_file}:force_style='{style}'",
        output_video
    ]
    print("Running FFmpeg command:")
    print(" ".join(command))
    subprocess.run(command, check=True)
    print(f"Subtitled video created: {output_video}")

def main():
    # Set your file paths here:
    input_video = "random clips/_M_For_Beyonce_Charlie_Kirk__Chris_Cuomo_Call_Out_Kamalas__Billion_Campaign_Spending_Spree__You_Part_5.mp4"           # Replace with your video file path.
    srt_file = "youtube_style_subtitles.srt"   # SRT file to be generated.
    output_video = "your_video_subbed.mp4"     # Final output video with burned-in subtitles.
    
    # Determine device (use CUDA if available; otherwise, use CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load the WhisperX transcription model.
    print("Loading WhisperX model...")
    pipeline = whisperx.load_model("medium", device, compute_type="float32")
    
    # Transcribe the video using WhisperX.
    print(f"Transcribing video: {input_video}")
    result = pipeline.transcribe(input_video)
    print("Transcription complete.")
    
    # Get the detected language (default to "en" if not found).
    language = result.get("language", "en")
    print(f"Detected language: {language}")
    
    # Load the alignment model and its metadata for the detected language.
    print("Loading alignment model...")
    alignment_model, metadata_align = whisperx.load_align_model(language, device)
    
    # Run forced alignment using the alignment model.
    print("Running forced alignment...")
    result_aligned = whisperx.align(result["segments"], alignment_model, metadata_align, input_video, device)
    print("Forced alignment complete.")
    
    # Generate an SRT file with YouTube-style (build-up) subtitles.
    print("Generating SRT file with build-up subtitles...")
    generate_srt_youtube_style(result_aligned["segments"], srt_file, threshold=5, overlap=2, max_gap=0.5)
    
    # Burn the generated subtitles into the video using FFmpeg.
    print("Burning subtitles into the video...")
    burn_subtitles(input_video, srt_file, output_video)

if __name__ == "__main__":
    main()
