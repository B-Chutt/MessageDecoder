import cv2
import os

def extract_frames(video_path, output_folder, frame_skip=1):
    """
    Extract frames from a video and save them as JPEG images.

    Parameters:
        video_path (str): Path to the input video file.
        output_folder (str): Folder where extracted frames will be saved.
        frame_skip (int): Save every nth frame (default = 1, saves all frames).
    """
    print("extract_frames running")

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Save every nth frame
        if frame_count % frame_skip == 0:
            filename = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
            print("Writing to " + filename)
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Done! Extracted {saved_count} frames to '{output_folder}'.")


def highlight_bright_areas(image_path, output_path, threshold=200):
    """
    Detect bright areas in a JPEG image and draw circles around them.

    Parameters:
        image_path (str): Path to the input JPEG image.
        output_path (str): Path to save the output image.
        threshold (int): Brightness threshold (0–255). Higher = fewer detections.
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image. Check the path.")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to find bright regions
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Find contours of bright regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw circles around bright areas
    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        if radius > 10:
            cv2.circle(img, center, radius, (0, 255, 0), 2)  # green circles
            print("LED at ", x, y)
        else:
            cv2.circle(img, center, radius, (0, 0, 255), 2)  # red circles
            print("Something at ", x, y)

    # Save result
    cv2.imwrite(output_path, img)
    print(f"Output saved to {output_path}")


def show_nth_frame_scaled(video_path, frame_number, target_width):
    """
    Read an MP4 file, extract the nth frame, and display it scaled to a given width.

    Parameters:
        video_path (str): Path to the MP4 video file.
        frame_number (int): The frame index to display (0-based).
        target_width (int): Desired width in pixels for the output frame.
    """
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Jump to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return

    # Get original dimensions
    h, w = frame.shape[:2]

    # Compute scaling factor to preserve aspect ratio
    scale = target_width / w
    target_height = int(h * scale)

    # Resize frame
    resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

    # Display the resized frame
    cv2.imshow(f"Frame {frame_number} (scaled)", resized_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cap.release()

# Example usage:
# show_nth_frame_scaled("input_video.mp4", 150, 640)  # shows frame 150 scaled to 640px width

print("message_decode running")
#extract_frames("message.mp4", "output_frames", frame_skip=30)
highlight_bright_areas("output_frames/frame_00001.jpg", "output_frames/frame_00001_marked.jpg", threshold=190)
show_nth_frame_scaled("message.mp4", 30, 500)
