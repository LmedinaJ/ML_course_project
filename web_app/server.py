import os
import glob
import re
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import rasterio
from PIL import Image
import numpy as np
from datetime import datetime
from collections import defaultdict

# Get absolute path to templates directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print('BASE_DIR',BASE_DIR)
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TIF_DIR = os.path.join(BASE_DIR, "tif_files")

# Create directories if they don't exist
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TIF_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Create a simple index.html file if it doesn't exist
index_html_path = os.path.join(TEMPLATES_DIR, "index.html")
with open(index_html_path, "w") as f:
    f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TIF File Viewer with Map</title>
    <!-- Leaflet CSS -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.css" />
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            line-height: 1.6;
        }
        .container {
            display: flex;
            height: 100vh;
        }
        .sidebar {
            width: 300px;
            padding: 15px;
            overflow-y: auto;
            border-right: 1px solid #ddd;
        }
        h1 {
            color: #333;
            margin-top: 0;
            font-size: 20px;
        }
        #map-container {
            flex-grow: 1;
        }
        #map {
            height: 100%;
            width: 100%;
        }
        .group {
            margin-bottom: 15px;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: #f9f9f9;
        }
        .timestamp {
            font-weight: bold;
            font-size: 16px;
            margin-bottom: 8px;
        }
        .files {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        .file-item {
            padding: 6px 10px;
            background-color: #e9e9e9;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        .file-item:hover {
            background-color: #d9d9d9;
        }
        .file-item.active {
            background-color: #4a80b9;
            color: white;
        }
        .controls {
            margin-top: 10px;
            display: flex;
            align-items: center;
        }
        button {
            padding: 6px 12px;
            background-color: #4a80b9;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin-right: 10px;
        }
        button:hover {
            background-color: #3a6fa8;
        }
        .frame-controls {
            margin-top: 10px;
            display: none;
        }
        .slider-container {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        #frame-slider {
            flex-grow: 1;
        }
        #frame-number {
            width: 30px;
            text-align: center;
        }
        .loading {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100px;
            font-style: italic;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h1>TIF File Viewer</h1>
            <button id="load-files">Load TIF Files</button>
            
            <div id="file-controls" style="display: none;">
                <div class="frame-controls" id="frame-controls">
                    <div class="slider-container">
                        <span>Frame:</span>
                        <input type="range" id="frame-slider" min="0" max="48" value="0">
                        <span id="frame-number">0</span>
                    </div>
                    <div class="slider-container">
                        <span>Opacity:</span>
                        <input type="range" id="opacity-slider" min="0" max="100" value="70">
                        <span id="opacity-value">70%</span>
                    </div>
                </div>
            </div>
            
            <div id="tif-groups"></div>
        </div>
        <div id="map-container">
            <div id="map"></div>
        </div>
    </div>

    <!-- Leaflet JS -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.js"></script>
    <script>
        // Initialize map
        const map = L.map('map').setView([13.7563, 100.5018], 6); // Center on Thailand

        // Add OpenStreetMap tile layer
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }).addTo(map);

        // Global variables
        let currentImageOverlay = null;
        let currentFilePath = null;
        let currentFrameIndex = 0;
        let totalFrames = 49; // Default
        let currentBounds = null;
        let tifGroups = [];

        // Load TIF files
        document.getElementById('load-files').addEventListener('click', async () => {
            try {
                document.getElementById('tif-groups').innerHTML = '<div class="loading">Loading TIF files...</div>';
                
                const response = await fetch('/list-tif-files');
                const data = await response.json();
                tifGroups = data.tif_groups;
                displayTifGroups(tifGroups);
                
                document.getElementById('file-controls').style.display = 'block';
            } catch (error) {
                console.error('Error loading TIF files:', error);
                document.getElementById('tif-groups').innerHTML = 
                    `<p style="color: red">Error loading TIF files: ${error.message}</p>`;
            }
        });

        // Display TIF groups in sidebar
        function displayTifGroups(groups) {
            const container = document.getElementById('tif-groups');
            container.innerHTML = '';
            
            if (groups.length === 0) {
                container.innerHTML = '<p>No TIF files found.</p>';
                return;
            }
            
            groups.forEach(group => {
                const groupDiv = document.createElement('div');
                groupDiv.className = 'group';
                
                const timestamp = document.createElement('div');
                timestamp.className = 'timestamp';
                timestamp.textContent = group.display_timestamp;
                groupDiv.appendChild(timestamp);
                
                const files = document.createElement('div');
                files.className = 'files';
                
                group.files.forEach(file => {
                    const fileItem = document.createElement('div');
                    fileItem.className = 'file-item';
                    fileItem.textContent = file.prefix + ' - ' + file.filename;
                    fileItem.dataset.framesDir = file.frames_dir;
                    fileItem.dataset.numFrames = file.num_frames;
                    fileItem.dataset.bounds = JSON.stringify(group.bounds);
                    
                    fileItem.addEventListener('click', () => {
                        // Deactivate all file items
                        document.querySelectorAll('.file-item').forEach(item => {
                            item.classList.remove('active');
                        });
                        
                        // Activate this file item
                        fileItem.classList.add('active');
                        
                        // Show frame controls
                        document.getElementById('frame-controls').style.display = 'block';
                        
                        // Update current file path and frames
                        currentFilePath = file.frames_dir;
                        totalFrames = file.num_frames;
                        
                        // Reset frame slider
                        const frameSlider = document.getElementById('frame-slider');
                        frameSlider.max = totalFrames - 1;
                        frameSlider.value = 0;
                        document.getElementById('frame-number').textContent = '0';
                        
                        // Parse bounds
                        currentBounds = JSON.parse(fileItem.dataset.bounds);
                        
                        // Display first frame
                        currentFrameIndex = 0;
                        displayFrame(currentFrameIndex);
                        
                        // Update map view to fit bounds
                        const leafletBounds = [
                            [currentBounds[0][0], currentBounds[0][1]], // [lat, lng] for southwest
                            [currentBounds[1][0], currentBounds[1][1]]  // [lat, lng] for northeast
                        ];
                        map.fitBounds(leafletBounds);
                    });
                    
                    files.appendChild(fileItem);
                });
                
                groupDiv.appendChild(files);
                container.appendChild(groupDiv);
            });
        }

        // Display a specific frame
        function displayFrame(frameIndex) {
            if (!currentFilePath || !currentBounds) return;
            
            // Remove existing overlay if any
            if (currentImageOverlay) {
                map.removeLayer(currentImageOverlay);
                currentImageOverlay = null;
            }
            
            // Create path to frame image
            const framePath = `${currentFilePath}/frame_${frameIndex}.png`;
            
            // Create image overlay with the current opacity
            const opacity = parseInt(document.getElementById('opacity-slider').value) / 100;
            
            // Define bounds for the image overlay
            const imageBounds = [
                [currentBounds[0][0], currentBounds[0][1]], // [lat, lng] for southwest
                [currentBounds[1][0], currentBounds[1][1]]  // [lat, lng] for northeast
            ];
            
            // Create and add image overlay
            currentImageOverlay = L.imageOverlay(framePath, imageBounds, {
                opacity: opacity
            }).addTo(map);
            
            // Update display
            document.getElementById('frame-number').textContent = frameIndex;
        }

        // Frame slider event
        document.getElementById('frame-slider').addEventListener('input', function() {
            currentFrameIndex = parseInt(this.value);
            displayFrame(currentFrameIndex);
        });

        // Opacity slider event
        document.getElementById('opacity-slider').addEventListener('input', function() {
            const opacity = parseInt(this.value);
            document.getElementById('opacity-value').textContent = `${opacity}%`;
            
            if (currentImageOverlay) {
                currentImageOverlay.setOpacity(opacity / 100);
            }
        });
    </script>
</body>
</html>""")

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (for PNG overlays)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    try:
        # Serve the index page
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        # Fallback to a simple HTML response if there's an issue with templates
        print(f"Error serving index template: {str(e)}")
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>TIF File Viewer</title>
        </head>
        <body>
            <h1>TIF File Viewer</h1>
            <p>Welcome to the TIF file viewer. <a href="/list-tif-files">View available TIF files</a>.</p>
        </body>
        </html>
        """)

@app.get("/list-tif-files")
async def list_tif_files():
    try:
        print(f"Scanning directory: {TIF_DIR}")
        all_files = glob.glob(os.path.join(TIF_DIR, "*"))
        print(f"All files in directory: {all_files}")
        tif_files = glob.glob(os.path.join(TIF_DIR, "*.tif"))
        print(f"Found {len(tif_files)} .tif files: {tif_files}")

        tif_groups = defaultdict(list)
        for tif_path in tif_files:
            filename = os.path.basename(tif_path)
            print(f"Processing file: {filename}")
            # Update regex to match uppercase 'B'
            match = re.match(r"(B[0-1][0-9])_(\d{8})_(\d{4})\.[tT][iI][fF]", filename)
            if not match:
                print(f"Skipping {filename}: does not match pattern")
                continue
            prefix = match.group(1)  # e.g., B03
            date_str = match.group(2)  # e.g., 20250420
            time_str = match.group(3)  # e.g., 1730
            timestamp = f"{date_str}_{time_str}"  # e.g., 20250420_1730
            tif_groups[timestamp].append((prefix, tif_path, filename))

        print(f"Grouped into {len(tif_groups)} timestamps: {list(tif_groups.keys())}")

        tif_info = []
        for timestamp, files in tif_groups.items():
            print(f"Processing timestamp {timestamp} with {len(files)} files")
            files.sort(key=lambda x: x[0])
            date_str, time_str = timestamp.split("_")
            display_timestamp = datetime.strptime(f"{date_str} {time_str}", "%Y%m%d %H%M").strftime("%Y-%m-%d %H:%M")

            group_files = []
            # Initialize bbox with default value
            bbox = [[12.0, 99.0], [15.0, 102.0]]  # Default bounds for Thailand
            valid_bounds_found = False
            
            for prefix, tif_path, filename in files:
                print(f"Processing {filename} for PNG generation")
                tif_static_dir = os.path.join(STATIC_DIR, filename.replace(".tif", "").replace(".TIF", ""))
                if not os.path.exists(tif_static_dir):
                    os.makedirs(tif_static_dir, exist_ok=True)
                    try:
                        with rasterio.open(tif_path) as src:
                            raster_data = src.read()
                            print(f"Shape of {filename}: {raster_data.shape}")
                            bounds = src.bounds
                            print(f"Bounds of {filename}: {bounds}")
                            # Only update bbox if valid bounds are found
                            if not (bounds.left == 0 and bounds.right == 0 and bounds.top == 0 and bounds.bottom == 0):
                                bbox = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
                                valid_bounds_found = True
                                print(f"Valid bounds found for {filename}, updating bbox to {bbox}")

                            for band in range(raster_data.shape[0]):
                                band_data = raster_data[band]
                                band_data_rgb = np.stack([band_data] * 3, axis=-1)
                                band_data_rgb = (band_data_rgb - band_data_rgb.min()) / (band_data_rgb.max() - band_data_rgb.min() + 1e-10) * 255
                                band_data_rgb = band_data_rgb.astype(np.uint8)
                                image = Image.fromarray(band_data_rgb)
                                png_path = os.path.join(tif_static_dir, f"frame_{band}.png")
                                image.save(png_path, "PNG")
                                print(f"Saved {png_path}")
                    except Exception as e:
                        print(f"Error processing {filename}: {str(e)}")
                        continue

                group_files.append({
                    "prefix": prefix,
                    "filename": filename,
                    "frames_dir": f"/static/{filename.replace('.tif', '').replace('.TIF', '')}",
                    "num_frames": 49
                })

            if group_files:
                tif_info.append({
                    "timestamp": timestamp,
                    "display_timestamp": display_timestamp,
                    "files": group_files,
                    "bounds": bbox  # Now this will always be defined
                })

        print(f"Returning {len(tif_info)} groups")
        return JSONResponse(content={"tif_groups": tif_info}, status_code=200)
    except Exception as e:
        print(f"Error in list_tif_files: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)