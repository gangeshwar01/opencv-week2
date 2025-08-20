# Project Report: Real-Time Multi-Stream RTSP Analysis

This report details the implementation of a multi-stream RTSP viewer with real-time motion detection and camera integrity checks. The project leverages OpenCV and Python's multithreading capabilities to process four concurrent video feeds.

---

### 1. Sources Consulted

The development process was guided by a combination of official documentation, tutorials, and community forums.

* **OpenCV Documentation:**
    * `cv2.VideoCapture`: Essential for understanding how to connect to and read from RTSP streams. The documentation on backend properties was particularly useful for debugging connection issues. ([docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html))
    * `cv2.createBackgroundSubtractorMOG2`: The primary source for implementing robust motion detection.
    * `cv2.Laplacian`: Key for implementing the blur detection algorithm. The tutorial on the Laplacian operator was consulted to understand its application for measuring edge sharpness. ([docs.opencv.org/4.x/d5/db5/tutorial_laplace_operator.html](https://docs.opencv.org/4.x/d5/db5/tutorial_laplace_operator.html))
    * `cv2.calcHist`: Used for analyzing pixel intensity distribution to detect covered lenses.

* **Python Documentation:**
    * `threading` Module: The official documentation was the main reference for creating, starting, and managing threads, and for using `threading.Lock` to ensure thread-safe access to shared data.

* **Community & Tutorials:**
    * **Stack Overflow:** Searched for common issues like "OpenCV VideoCapture RTSP lag" and "fast blur detection python opencv". These forums provided practical thresholds and optimization tips, such as the Laplacian variance threshold of 100.
    * **PyImageSearch Blog:** Adrian Rosebrock's articles on motion detection and blur detection provided foundational concepts and code examples that were adapted for this project.

---

### 2. Key Learnings, Insights, and Conclusions

#### **Task 1: Multi-Stream RTSP Viewer**

* **Challenge: RTSP Stream Reliability:** The biggest initial hurdle was finding stable, public RTSP streams. Many are offline, region-locked, or have high latency.
* **Insight:** Using multithreading is crucial. Each stream's I/O operations (waiting for network packets) can block, but running each in a separate thread allows the main application to remain responsive. A "producer-consumer" pattern was implemented, where reader threads "produce" frames and the main thread "consumes" them for processing.
* **Learning:** A shared dictionary protected by a `threading.Lock` is a simple yet effective mechanism for passing frames from worker threads to the main thread. Acquiring the lock for the briefest possible moment (just to write/read the frame) is key to minimizing contention.

#### **Task 2: Real-Time Motion Detection**

* **Challenge: Performance vs. Accuracy:** The initial implementation of motion detection caused a noticeable drop in FPS.
* **Learning:** `cv2.createBackgroundSubtractorMOG2` is powerful but computationally more expensive than simple frame differencing. However, it's far more resilient to gradual lighting changes. For this application, its robustness was worth the performance cost, which was mitigated through other optimizations.
* **Optimization:** The most significant optimization was resizing frames *immediately* after they are read in the worker thread. Processing smaller frames (640x480) dramatically improved the performance of all subsequent steps (background subtraction, integrity checks) and allowed the application to run in real-time.

#### **Task 3: Camera Integrity Check**

* **Challenge: Defining "Compromised":** The definitions for blur, coverage, and laser effects are heuristic and required experimentation to find suitable thresholds.
* **Insight (Blur Detection):** The variance of the Laplacian is an incredibly fast and effective metric for blur. A single, well-chosen threshold (`<100`) worked reliably across different test videos. It quantifies the presence of edges, which are sparse in blurry images.
* **Insight (Coverage Detection):** Analyzing the histogram is a robust way to detect a lack of visual information. If a camera is covered (black screen) or blinded by a laser/spotlight (white screen), the histogram will show one dominant color bin. This method cleverly handles multiple failure modes with a single check.
* **Conclusion:** These integrity checks are effective as first-level alerts but could be fooled. For instance, a plain, uniformly lit wall might trigger the coverage alert. A production system would need more sophisticated logic, such as analyzing changes over time.

---

### 3. Practice Attempts and Exercises

1.  **Single-Stream Script:** The first step was a simple script to connect to a single RTSP stream and display it. This helped resolve initial codec and connection issues.
2.  **Blur Threshold Tuning:** A separate script was created to load a sharp image and its blurred version. I printed the Laplacian variance for both to find a reliable threshold that could distinguish between them.
3.  **Threading without Locks:** An early attempt did not use a `threading.Lock`. This led to occasional visual artifacts and crashes, demonstrating the importance of thread safety when accessing shared memory.
4.  **Grid Assembly Logic:** Initially, I tried to build the grid conditionally, which was complex. A simpler approach was adopted: create a list of all processed frames (or blank placeholders if a stream is down) and then use `np.hstack` and `np.vstack` on slices of that list. This made the code cleaner and more robust.
