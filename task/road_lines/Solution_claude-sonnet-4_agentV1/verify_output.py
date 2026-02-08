import cv2
import numpy as np

def extract_sample_frames(video_path, output_prefix, frame_indices):
    """Extract sample frames from video for verification"""
    cap = cv2.VideoCapture(video_path)

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            output_path = f"{output_prefix}_frame_{frame_idx}.jpg"
            cv2.imwrite(output_path, frame)
            print(f"Saved frame {frame_idx} to {output_path}")
        else:
            print(f"Failed to read frame {frame_idx}")

    cap.release()
    print(f"\nExtracted {len(frame_indices)} sample frames")

def verify_video_properties(video_path):
    """Verify output video properties"""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"ERROR: Cannot open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {video_path}")
    print(f"FPS: {fps}")
    print(f"Resolution: {width}x{height}")
    print(f"Total Frames: {frame_count}")
    print(f"Duration: {frame_count/fps:.2f} seconds")

    cap.release()
    return frame_count

if __name__ == "__main__":
    print("=== Verifying Output Video ===\n")
    frame_count = verify_video_properties("output.mp4")

    print("\n=== Extracting Sample Frames ===\n")
    # Extract frames from different parts of the video
    sample_indices = [
        100,   # Early frame (likely no detection)
        700,   # First detection area
        1500,  # Mid video
        3000,  # Later section
        5000   # Near end
    ]
    extract_sample_frames("output.mp4", "sample", sample_indices)

    print("\n=== Verification Complete ===")
