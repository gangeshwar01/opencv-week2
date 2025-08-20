# code/main.py

import cv2
import threading
import numpy as np
import time
from collections import deque

# --- Configuration ---
# NOTE: Public RTSP streams are unstable. These may be offline.
# Find more at sites like https://www.insecam.org/
# Using public test streams for demonstration purposes.
RTSP_STREAMS = [
    # Big Buck Bunny - A reliable test stream
    "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov",
    # A second instance of the same stream for demonstration
    "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov",
    # A third instance
    "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov",
    # A fourth instance
    "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov"
]

# Frame settings
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
GRID_SIZE = (2, 2)

# Motion Detection settings
MOTION_CONTOUR_AREA = 1000 # Min area to be considered motion

# Camera Integrity settings
BLUR_THRESHOLD = 100.0  # Variance of Laplacian threshold. Lower is more blurry.
COVERAGE_THRESHOLD_PCT = 0.75 # Pct of frame dominated by one color

# --- Global Variables ---
# A thread-safe dictionary to hold the latest frame from each stream
frames = {}
frame_lock = threading.Lock()
stop_threads = False

# --- Stream Reader Thread ---
def stream_reader(stream_index, stream_url):
    """
    A function to be run in a thread that reads frames from an RTSP stream.
    """
    global frames, stop_threads

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print(f"Error: Could not open stream {stream_index} at {stream_url}")
        return

    print(f"Stream {stream_index} started successfully.")

    while not stop_threads:
        ret, frame = cap.read()
        if not ret:
            print(f"Stream {stream_index} ended or failed to read frame.")
            # Optional: Add a retry mechanism here
            time.sleep(1) # Wait before retrying
            cap.release()
            cap.open(stream_url)
            continue
        
        # Resize frame for uniform grid display
        resized_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        
        with frame_lock:
            frames[stream_index] = resized_frame.copy()

    cap.release()
    print(f"Stream {stream_index} stopped.")

# --- Integrity Check Functions ---
def check_blur(frame, threshold):
    """
    Checks if a frame is blurry by calculating the variance of the Laplacian.
    A low variance suggests a lack of detail and edges, indicating blur.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

def check_coverage(frame, threshold_pct=0.75):
    """
    Checks if the camera is covered or affected by a laser by analyzing the color histogram.
    If a single color dominates > threshold_pct of the frame, it's considered compromised.
    """
    # Using grayscale histogram for simplicity to detect uniform color or over/under exposure
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    total_pixels = frame.shape[0] * frame.shape[1]
    
    # Find the peak in the histogram
    max_val = hist.max()
    
    # If the peak value accounts for more than the threshold percentage of pixels
    if max_val / total_pixels > threshold_pct:
        return True
    return False

# --- Drawing Utility ---
def draw_text_with_background(frame, text, position, font_scale=0.7, color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Draws text with a solid background for better visibility."""
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    top_left = position
    bottom_right = (top_left[0] + text_width + 10, top_left[1] + text_height + 10)
    cv2.rectangle(frame, top_left, bottom_right, bg_color, -1)
    cv2.putText(frame, text, (top_left[0] + 5, top_left[1] + text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)


# --- Main Application Logic ---
if __name__ == "__main__":
    # Start a thread for each RTSP stream
    threads = []
    for i, url in enumerate(RTSP_STREAMS):
        thread = threading.Thread(target=stream_reader, args=(i, url), daemon=True)
        threads.append(thread)
        thread.start()

    # Background subtractor for motion detection
    bg_subtractors = [cv2.createBackgroundSubtractorMOG2() for _ in RTSP_STREAMS]

    # Placeholder for a blank frame if a stream is not available
    blank_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    cv2.putText(blank_frame, "No Signal", (50, FRAME_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    print("Main thread started. Press 'q' to quit.")

    while True:
        start_time = time.perf_counter()
        
        processed_frames = []
        with frame_lock:
            # Create a copy to avoid holding the lock during processing
            current_frames = frames.copy()

        for i in range(len(RTSP_STREAMS)):
            frame = current_frames.get(i, blank_frame.copy())
            
            # --- 1. Motion Detection ---
            fg_mask = bg_subtractors[i].apply(frame)
            # Apply thresholding to remove noise/shadows
            _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            motion_detected = False
            for contour in contours:
                if cv2.contourArea(contour) > MOTION_CONTOUR_AREA:
                    motion_detected = True
                    break # No need to check other contours
            
            if motion_detected:
                draw_text_with_background(frame, "Motion Detected", (10, 10), bg_color=(0, 0, 255))
        
            # --- 2. Camera Integrity Check ---
            is_blurred = check_blur(frame, BLUR_THRESHOLD)
            is_covered = check_coverage(frame, COVERAGE_THRESHOLD_PCT)
            
            if is_blurred or is_covered:
                reason = "Blurred" if is_blurred else "Covered/Laser"
                warning_text = f"Camera Compromised: {reason}"
                draw_text_with_background(frame, warning_text, (10, 50), bg_color=(0, 215, 255), color=(0,0,0))
            
            processed_frames.append(frame)

        # --- 3. Assemble Grid ---
        rows = []
        for i in range(GRID_SIZE[0]):
            start_index = i * GRID_SIZE[1]
            end_index = start_index + GRID_SIZE[1]
            row = np.hstack(processed_frames[start_index:end_index])
            rows.append(row)
        
        grid_frame = np.vstack(rows)

        # --- Display FPS ---
        end_time = time.perf_counter()
        fps = 1 / (end_time - start_time)
        cv2.putText(grid_frame, f"FPS: {fps:.2f}", (10, grid_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show the final combined frame
        cv2.imshow("Multi-Stream RTSP Viewer", grid_frame)

        # Check for 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    print("Shutting down...")
    stop_threads = True
    for thread in threads:
        thread.join() # Wait for all threads to finish

    cv2.destroyAllWindows()
    print("Application closed.")
