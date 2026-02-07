import cv2
import os
import math
from collections import defaultdict

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


def get_lit_led_positions(frame):
    positions = []

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold to find bright regions
    _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

    # Find contours of bright regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw circles around bright areas
    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        if (radius > 10) and (y < 1420):
            positions.append((x, y))
    return positions

def get_led_positions(video_path):
    all_positions = []

    # Open video file
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        print("Error: Could not open video.")
        return
    total_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_rate = vid.get(cv2.CAP_PROP_FPS)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("Video file has " + str(total_frames) + " frames and runs at " + str(frame_rate)+" FPS")
    print("Frame size is " + str(width) + " X " + str(height)+".")

    got_frame = True
    frame_number = 0

    while got_frame:
        got_frame, frame = vid.read()
        if got_frame:
            found_positions = get_lit_led_positions(frame)
            print("Frame " + str(frame_number) + ". " + str(len(found_positions)) + " lit LEDs found")
            if frame_number == 47:
                print("It's about to go wrong")
            new_led_count = 0
            for found_pos in found_positions:
                already_seen = 0
                for existing_pos in all_positions:
                    if (abs(found_pos[0] - existing_pos[0]) < 10) and (abs(found_pos[1] - existing_pos[1]) < 10):
                        already_seen += 1
                        if already_seen > 1:
                            print("ERROR LED stored multiple times")
                if already_seen == 0:
                    all_positions.append(found_pos)
                    new_led_count += 1
            if new_led_count != 0:
                print("Found " + str(new_led_count) + " new LEDs")
            frame_number += 1
    vid.release()
    return all_positions


def split_and_sort_hexagon_sides(points):
    """
    Given a list of (x, y) points that lie roughly on the sides of a hexagon,
    return six lists of points, each sorted anti-clockwise along the hexagon.
    """

    if not points:
        return [[] for _ in range(6)]

    # --- 1. Compute centroid ---
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)

    # --- 2. Compute angle of each point relative to centroid ---
    angle_data = []
    for p in points:
        dx = p[0] - cx
        dy = p[1] - cy
        ang = math.atan2(dy, dx)
        ang += (math.pi/6) #Offset by half a side
        ang = (ang + 2*math.pi) % (2*math.pi)
        angle_data.append((p, ang))

    # --- 3. Assign to 6 angular sectors ---
    sector_size = 2 * math.pi / 6
    sides = defaultdict(list)
    for p, ang in angle_data:
        sector = int(ang // sector_size)
        sides[sector].append((p, ang))

    # --- 4. Sort the sides themselves anti-clockwise ---
    sorted_sides = []
    for i in range(6):
        side = sides[i]

        # Sort points within each side by angle (anti-clockwise)
        side_sorted = sorted(side, key=lambda x: x[1])

        # Keep only the points, not the angles
        sorted_sides.append([p for p, _ in side_sorted])

    return sorted_sides


def get_frame_data(coords, frame):
    # Convert to grayscale
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    output_bytes = []
    for side in coords:
    #side = coords[2]
    #if True:
        value = 0
        bit_weight = 1
        for led in side:
            if grey[int(led[1]), int(led[0])] > 220:
                value += bit_weight
#                cv2.circle(frame, (int(led[0]),int(led[1])), 25, (0, 255, 0), 2)  # green circles
#            else:
#                cv2.circle(frame, (int(led[0]),int(led[1])), 25, (255, 0, 0), 2)  # red circles
            bit_weight *= 2
        output_bytes.append(value)
#    resized_frame = cv2.resize(frame, (540, 960), interpolation=cv2.INTER_AREA)
#    cv2.imshow(f"How about this?", resized_frame)
    return output_bytes

def get_all_data(coords, video_path):
    all_data = []
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        print("Error: Could not open video.")
    
    # Look for pattern changes. Need to account for the fact that a pattern change normally results in at-least one frame of rubbbish 
    # 
    # To be taken as a new pattern, we need to see a change from one
    # frame to the next, followed by a number of identical frames in a row
    
    PATTERN_STABLE = 0
    CHANGE_IN_PROGRESS = 1


    stable_count = 0
    state = PATTERN_STABLE 
    got_frame, frame = vid.read()
    if got_frame:
        last_stable_frame = get_frame_data(sides, frame)
        all_data.append(last_stable_frame)
    while got_frame:
        new_frame_data = get_frame_data(sides, frame)
        if state == PATTERN_STABLE:
            if new_frame_data != last_stable_frame:
                stable_count = 0
                last_frame_data = new_frame_data
                state = CHANGE_IN_PROGRESS
        elif state == CHANGE_IN_PROGRESS:
            if new_frame_data == last_frame_data:
                stable_count += 1
                if stable_count > 3:
                    all_data.append(last_frame_data)
                    last_stable_frame = last_frame_data
                    state = PATTERN_STABLE
            else:
                stable_count = 0
                last_frame_data = new_frame_data
        got_frame, frame = vid.read()
    vid.release()
    return all_data

def rotate_sides(raw_data, starting_point):
    rotated_all = []
    for hexagon in raw_data:
        rotated_hexagon = [0, 0, 0, 0, 0, 0]
        side_number = starting_point
        for byte in hexagon:
            rotated_hexagon[side_number] = byte
            side_number += 1
            if side_number >= 6:
                side_number = 0
        rotated_all.append(rotated_hexagon)
    return rotated_all

print("message_decode running")
leds = get_led_positions("message.mp4")
sides = split_and_sort_hexagon_sides(leds)
for i in range(6):
    print("\r\nSide " + str(i) + ".")
    for led in sides[i]:
        print (str(led[0]) + "," + str(led[1]))
    print("")
raw_data = get_all_data(sides, "message.mp4")
print("Raw message data")


for rotation in range(1,6):
    rotated_data = rotate_sides(raw_data,rotation)
    for hexagon in rotated_data:
        for byte in hexagon:
            print(chr(byte), end='')
    print("")




#extract_frames("message.mp4", "output_frames", frame_skip=30)
#highlight_bright_areas("output_frames/frame_00001.jpg", "output_frames/frame_00001_marked.jpg", threshold=190)
#show_nth_frame_scaled("message.mp4", 30, 500)
