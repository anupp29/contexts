import ffmpeg

def compress_video(input_path, output_path, target_crf=24):
    """
    Compresses a video file using H.264 codec and CRF control.
    
    Parameters:
    - input_path: Path to the 1.3GB source video file.
    - output_path: Path where the compressed video will be saved.
    - target_crf: Quality metric (18-28). Higher number = smaller size, lower quality.
    """
    try:
        print("Starting video compression... This will take a few minutes for a 1.3GB file.")
        
        (
            ffmpeg
            .input(input_path)
            .output(
                output_path, 
                vcodec='libx264',    # Standard, highly compatible H.264 codec
                crf=target_crf,       # Controls quality/file size balance
                acodec='aac',         # Compresses audio efficiently
                audio_bitrate='128k'  # Saves space on audio tracks
            )
            .overwrite_output()
            .run()
        )
        
        print(f"Success! Compressed video saved to: {output_path}")
        
    except ffmpeg.Error as e:
        print("An error occurred during compression:", e.stderr.decode())

# Example usage:
compress_video("C:/Users/hp/Videos/Captures/Meet - men-sapi-nbm - Google Chrome 2026-07-25 19-04-59.mp4", "compressed_output.mp4", target_crf=24)
