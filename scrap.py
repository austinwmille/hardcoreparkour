import os
import subprocess
from datetime import timedelta
import torch
import whisperx

#############################
# Helper Functions
#############################

def format_ass_timestamp(seconds):
    """
    Convert a float number of seconds into an ASS timestamp (H:MM:SS.cs).
    """
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    centiseconds = int((seconds - total_seconds) * 100)
    return f"{hours}:{minutes:02}:{secs:02}.{centiseconds:02}"

def generate_ass_karaoke_subtitles(aligned_segments, ass_path, max_words=3, pause_threshold=0.3):
    """
    Generate an ASS subtitle file using karaoke timing.
    
    The algorithm groups word-level alignments into blocks. A new block is started when either:
      - The block reaches max_words (default 3), or
      - The gap between consecutive words is greater than pause_threshold seconds.
    
    For each block:
      - The dialogue start time is the start time of the first word in the block.
      - The dialogue end time is the end time of the last word.
      - For each word in the block, we compute a karaoke duration in centiseconds:
          For non-last words: (start_time of next word - start_time of current word)*100.
          For the last word: (end_time - start_time)*100.
      - Each word is prefixed with its karaoke tag and an override tag to "pop":
            {\t(0,200,\fscx150,\fscy150)}{\kXX}word
        which animates the word scaling to 150% over 200ms before settling.
    """
    # ASS header with style.
    header = (
        "[Script Info]\n"
        "Title: Karaoke Subtitles\n"
        "ScriptType: v4.00+\n"
        "Collisions: Normal\n"
        "PlayResX: 1280\n"
        "PlayResY: 720\n"
        "Timer: 100.0000\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,"
        "Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
        "Alignment,MarginL,MarginR,MarginV,Encoding\n"
        # Using Arial, size 36, primary colour = #6A0DAD (ASS BGR: &H00AD0D6A), no background,
        # Outline 1, Shadow 2, Alignment=2 (bottom-center) and MarginV=80 (raising the subs higher).
        "Style: Default,Arial,36,&H00AD0D6A,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,2,2,10,10,300,1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    
    # First, flatten all words from all segments.
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
    # Sort words by start time.
    words_list.sort(key=lambda x: x["start"])
    
    # Group words into blocks.
    blocks = []
    current_block = []
    for i, w in enumerate(words_list):
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
    
    # Write the ASS file.
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
                # Compute duration in centiseconds.
                if i < n - 1:
                    duration_cs = int(round((block[i+1]["start"] - word["start"]) * 100))
                else:
                    duration_cs = int(round((word["end"] - word["start"]) * 100))
                # Add pop effect override tag; adjust values as desired.
                karaoke_text += f"{{\\t(0,200,\\fscx150,\\fscy150)}}{{\\k{duration_cs}}}{word_text} "
            karaoke_text = karaoke_text.strip()
            line = (
                f"Dialogue: 0,{format_ass_timestamp(block_start)},{format_ass_timestamp(block_end)},Default,,0,0,0,,{karaoke_text}\n"
            )
            f.write(line)
    print(f"ASS karaoke subtitle file generated: {ass_path}")

def burn_subtitles(input_video, ass_file, output_video):
    """
    Burn ASS subtitles into the video using FFmpeg.
    """
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

#############################
# Main Script Flow
#############################

def main():
    # Set your file paths here:
    input_video = "random clips/_M_For_Beyonce_Charlie_Kirk__Chris_Cuomo_Call_Out_Kamalas__Billion_Campaign_Spending_Spree__You_Part_5.mp4"  # Replace with your video file path.
    ass_file = "karaoke_subtitles.ass"         # ASS subtitle file to be generated.
    output_video = "your_video_subbed.mp4"       # Final output video with burned-in subtitles.
    
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
    
    # Generate the ASS karaoke subtitle file with the new grouping and styling.
    print("Generating ASS karaoke subtitle file...")
    generate_ass_karaoke_subtitles(result_aligned["segments"], ass_file, max_words=3, pause_threshold=0.3)
    
    # Burn the generated ASS subtitles into the video using FFmpeg.
    print("Burning subtitles into the video...")
    burn_subtitles(input_video, ass_file, output_video)

if __name__ == "__main__":
    main()
