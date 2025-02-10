import os
import subprocess
from datetime import timedelta
import torch
import whisperx

def format_timestamp(seconds):
    """
    Convert a float number of seconds into an SRT timestamp (HH:MM:SS,ms)
    """
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    milliseconds = int((seconds - total_seconds) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

def generate_srt_whisperx(aligned_segments, srt_path):
    """
    Generate an SRT file using word-level timestamps from WhisperX's aligned segments.
    For each segment, we iterate through its words and create a sliding window (up to 3 words).
    """
    block_number = 1
    with open(srt_path, "w", encoding="utf-8") as f:
        for segment in aligned_segments:
            if "words" in segment:
                words = segment["words"]
                for i, word_info in enumerate(words):
                    block_start = word_info["start"]
                    block_end = word_info["end"]
                    # Create a sliding window of words:
                    # For the first two words, show all words so far.
                    # From the third word onward, show the previous two words plus the current word.
                    if i < 2:
                        window_words = [w["word"] for w in words[:i+1]]
                    else:
                        window_words = [w["word"] for w in words[i-2:i+1]]
                    text_line = " ".join(window_words)
                    f.write(f"{block_number}\n")
                    f.write(f"{format_timestamp(block_start)} --> {format_timestamp(block_end)}\n")
                    f.write(f"{text_line}\n\n")
                    block_number += 1
    print(f"Word-level SRT file generated: {srt_path}")

def burn_subtitles(input_video, srt_file, output_video):
    """
    Burn subtitles into the video using FFmpeg.
    The force_style option sets Alignment=8 (top center) and MarginV=100 to push the subtitles down.
    Adjust these values if needed.
    """
    command = [
        "ffmpeg",
        "-y",  # Overwrite output if it exists
        "-i", input_video,
        "-vf", f"subtitles={srt_file}:force_style='Alignment=8,MarginV=100'",
        output_video
    ]
    print("Running FFmpeg command:")
    print(" ".join(command))
    subprocess.run(command, check=True)
    print(f"Subtitled video created: {output_video}")

def main():
    # Set your file paths here:
    input_video = "./random clips/_M_For_Beyonce_Charlie_Kirk__Chris_Cuomo_Call_Out_Kamalas__Billion_Campaign_Spending_Spree__You_Part_5"           # Replace with your video file path.
    srt_file = "whisperx_subtitles.srt"        # Path for the generated SRT file.
    output_video = "your_video_subbed.mp4"     # Output video with burned-in subtitles.
    
    # Determine device (use CUDA if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load the WhisperX model and alignment metadata.
    print("Loading WhisperX model...")
    model, metadata = whisperx.load_model("medium", device)
    
    # Transcribe the video using WhisperX.
    print(f"Transcribing video: {input_video}")
    result = model.transcribe(input_video)
    print("Transcription complete.")
    
    # Run forced alignment via WhisperX.
    # Note: The alignment function expects: (segments, model, metadata, audio_file, device).
    # Here we pass the input_video as the audio file (WhisperX can extract the audio internally).
    print("Running forced alignment...")
    result_aligned = whisperx.align(result["segments"], model, metadata, input_video, device)
    print("Forced alignment complete.")
    
    # Generate an SRT file with word-level timestamps.
    generate_srt_whisperx(result_aligned["segments"], srt_file)
    
    # Burn the generated subtitles into the video.
    burn_subtitles(input_video, srt_file, output_video)

if __name__ == "__main__":
    main()
