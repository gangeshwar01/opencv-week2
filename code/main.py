# code/main.py
import cv2
import threading
import numpy as np
import time

# --- Configuration ---
# NOTE: Public RTSP streams are unstable. These may be offline. Try commented http links
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

# # --- Configuration ---
# RTSP_STREAMS = [
#     # Replace with your RTSP camera URLs
#     "http://61.211.241.239/nphMotionJpeg?Resolution=320x240&Quality=Standard",
#     "http://47.51.131.147/-wvhttp-01-/GetOneShot?image_size=1280x720&frame_count=1000000000",
#     "http://takemotopiano.aa1.netvolante.jp:8190/nphMotionJpeg?Resolution=640x480&Quality=Standard&Framerate=30",
#     "http://pendelcam.kip.uni-heidelberg.de/mjpg/video.mjpg"
# ]

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
GRID_SIZE = (2, 2)

MOTION_CONTOUR_AREA = 1000
BLUR_THRESHOLD = 100.0
COVERAGE_THRESHOLD_PCT = 0.75

frames = {}                 # {index: (frame, is_blank)}
frame_lock = threading.Lock()
stop_threads = False

# --- Stream Reader ---
def stream_reader(stream_index, stream_url):
    """
    Thread function that connects to an RTSP stream, reads frames,
    and auto-reconnects if the stream fails.
    """
    global frames, stop_threads
    cap = None
    retry_delay = 1.0

    while not stop_threads:
        if cap is None or not cap.isOpened():
            if cap:
                cap.release()
            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                print(f"[Stream {stream_index}] Failed to open. Retrying in {retry_delay:.1f}s...")
                with frame_lock:
                    frames[stream_index] = (blank_frame.copy(), True)  # mark as blank
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 10)
                continue
            else:
                print(f"[Stream {stream_index}] Connected.")
                retry_delay = 1.0

        ret, frame = cap.read()
        if not ret:
            print(f"[Stream {stream_index}] Frame read failed. Reconnecting...")
            cap.release()
            cap = None
            with frame_lock:
                frames[stream_index] = (blank_frame.copy(), True)
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 10)
            continue

        retry_delay = 1.0
        resized_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        with frame_lock:
            frames[stream_index] = (resized_frame, False)  # real frame

    if cap:
        cap.release()
    print(f"[Stream {stream_index}] Thread stopped.")

# --- Integrity Check Functions ---
def check_blur(frame, threshold):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

def check_coverage(frame, threshold_pct=0.75):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    total_pixels = frame.shape[0] * frame.shape[1]
    return (hist.max() / total_pixels) > threshold_pct

# --- Drawing Utility ---
def draw_text_with_background(frame, text, position, font_scale=0.7,
                              color=(255, 255, 255), bg_color=(0, 0, 0)):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    tl = position
    br = (tl[0] + tw + 10, tl[1] + th + 10)
    cv2.rectangle(frame, tl, br, bg_color, -1)
    cv2.putText(frame, text, (tl[0] + 5, tl[1] + th + 5), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, 2)

# --- Main Application ---
if __name__ == "__main__":
    # Placeholder frame
    blank_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    cv2.putText(blank_frame, "No Signal", (50, FRAME_HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Start stream threads
    threads = []
    for i, url in enumerate(RTSP_STREAMS):
        t = threading.Thread(target=stream_reader, args=(i, url), daemon=True)
        threads.append(t)
        t.start()

    bg_subtractors = [cv2.createBackgroundSubtractorMOG2() for _ in RTSP_STREAMS]
    cells_needed = GRID_SIZE[0] * GRID_SIZE[1]

    print("[Main] Started. Press 'q' to quit.")

    try:
        while True:
            start_time = time.perf_counter()
            with frame_lock:
                current_frames = frames.copy()

            processed_frames = []
            for i in range(len(RTSP_STREAMS)):
                frame, is_blank = current_frames.get(i, (blank_frame.copy(), True))

                # Motion detection only on real frames
                if not is_blank:
                    fg_mask = bg_subtractors[i].apply(frame)
                    _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
                    res = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours = res[0] if len(res) == 2 else res[1]
                    if any(cv2.contourArea(c) > MOTION_CONTOUR_AREA for c in contours):
                        draw_text_with_background(frame, "Motion Detected", (10, 10), bg_color=(0, 0, 255))

                    # Integrity checks
                    if check_blur(frame, BLUR_THRESHOLD):
                        draw_text_with_background(frame, "Camera Compromised: Blurred", (10, 50),
                                                  bg_color=(0, 215, 255), color=(0, 0, 0))
                    elif check_coverage(frame, COVERAGE_THRESHOLD_PCT):
                        draw_text_with_background(frame, "Camera Compromised: Covered/Laser", (10, 50),
                                                  bg_color=(0, 215, 255), color=(0, 0, 0))

                processed_frames.append(frame)

            # Pad/truncate to fill grid
            if len(processed_frames) < cells_needed:
                processed_frames += [blank_frame.copy()] * (cells_needed - len(processed_frames))
            elif len(processed_frames) > cells_needed:
                processed_frames = processed_frames[:cells_needed]

            # Assemble grid
            rows = []
            for r in range(GRID_SIZE[0]):
                s, e = r * GRID_SIZE[1], (r + 1) * GRID_SIZE[1]
                rows.append(np.hstack(processed_frames[s:e]))
            grid_frame = np.vstack(rows)

            # FPS
            dt = time.perf_counter() - start_time
            fps = 1.0 / dt if dt > 1e-6 else 0.0
            cv2.putText(grid_frame, f"FPS: {fps:.2f}",
                        (10, grid_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Multi-Stream RTSP Viewer", grid_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("[Main] Shutting down...")
        stop_threads = True
        for t in threads:
            t.join()
        cv2.destroyAllWindows()
        print("[Main] Closed.")
