# opencv-week2
Real-Time Multi-Stream RTSP Viewer & Analyzer
This project is a high-performance, multi-stream RTSP viewer built with Python, OpenCV, and multithreading. It concurrently processes four video feeds, displaying them in a 2x2 grid while performing real-time motion detection and camera integrity checks on each stream.

(This is a representative image. Replace with a screenshot from your diagrams/ folder.)

✨ Features
Multi-Stream Viewer: Simultaneously connects to and displays up to four RTSP streams in a 2x2 grid.

Multithreaded Architecture: Uses a dedicated thread for each stream to prevent I/O blocking and ensure a smooth, responsive UI.

Real-Time Motion Detection: Implements background subtraction (cv2.BackgroundSubtractorMOG2) to identify and flag motion in each feed.

Camera Integrity Checks:

Blur Detection: Uses the variance of the Laplacian to detect out-of-focus or blurry camera feeds.

Coverage/Tamper Detection: Analyzes the color histogram to identify covered lenses or feeds blinded by a laser or bright light.

Performance Monitoring: Displays the real-time processing Frames Per Second (FPS) of the main application loop.

📂 Project Structure
The repository is organized to separate code, documentation, and visual assets.

opencv-week2/
├── code/
│   └── main.py              # Main application script
├── diagrams/
│   ├── final_output.png     # Screenshot of the running application
│   └── flow_diagram.png     # Architectural flow diagram
├── report/
│   └── report.md            # Detailed project report
├── .gitignore               # Standard Python gitignore
├── requirements.txt         # Project dependencies
└── README.md                # You are here!
⚙️ Setup and Installation
Follow these steps to get the project running on your local machine.

Prerequisites
Python 3.8+

pip (Python package installer)

Installation Steps
Clone the repository:

Bash

git clone https://github.com/YOUR_USERNAME/opencv-week2.git
cd opencv-week2
Create a virtual environment (recommended):

Bash

# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
Install the required packages:

Bash

pip install -r requirements.txt
▶️ How to Run
Navigate to the code directory:

Bash

cd code
Run the main script:

Bash

python main.py
The application window will open, displaying the four video streams. Press the q key to close the application gracefully.

Note: The default RTSP streams in main.py are public test streams and may be unstable or offline. You can replace them with your own RTSP URLs by editing the RTSP_STREAMS list at the top of the code/main.py file.

🏛️ System Architecture
The application uses a producer-consumer model. Each "producer" is a worker thread responsible for capturing frames from a single RTSP stream. The "consumer" is the main thread, which reads the latest frames, performs analysis, assembles the grid, and displays the final output. This decouples network I/O from video processing, leading to better performance.
