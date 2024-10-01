
from flask import Flask, request, jsonify, render_template, send_from_directory
import cv2
import mediapipe as mp
import numpy as np
import subprocess
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Serve files from the upload folder
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# Initialize MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

def process_video(video_path, background_path):
    # Extract audio using ffmpeg
    audio_output_path = os.path.join(UPLOAD_FOLDER, "audio.aac")
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'aac', audio_output_path])

    # Initialize video capture with the uploaded video file
    cap = cv2.VideoCapture(video_path)

    # Check if video capture opened successfully
    if not cap.isOpened():
        raise Exception("Error: Could not open video file.")

    # Get video properties (width, height, fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0 or np.isnan(fps):
        fps = 25  # Default FPS

    # Define video writer to save the output (MP4 format)
    video_output_path = os.path.join(UPLOAD_FOLDER, 'output_with_custom_background.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

    # Load the background image and resize it to match video dimensions
    background_img = cv2.imread(background_path)
    background_img = cv2.resize(background_img, (width, height))

    # Processing video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform segmentation using MediaPipe
        result = segmentation.process(frame_rgb)

        # Get the segmentation mask
        mask = result.segmentation_mask

        # Threshold the mask to make it binary (foreground-background separation)
        mask = np.where(mask > 0.5, 1, 0).astype(np.uint8)

        # Apply the mask to the original frame to extract the foreground
        fg_mask = mask  # Foreground mask
        bg_mask = 1 - fg_mask  # Background mask (inverted)

        # Extract the foreground from the original frame
        fg = cv2.bitwise_and(frame, frame, mask=fg_mask)

        # Apply the background image to the regions where the background is detected
        bg = cv2.bitwise_and(background_img, background_img, mask=bg_mask)

        # Combine the foreground and the background
        result_frame = cv2.add(fg, bg)

        # Write the result frame to the output video
        out.write(result_frame)

    cap.release()
    out.release()

    # Merge the audio with the processed video using ffmpeg
    final_output_path = os.path.join(UPLOAD_FOLDER, 'final_output_with_audio_and_custom_background.mp4')
    subprocess.run([
        'ffmpeg',
        '-y', 
        '-i', video_output_path, 
        '-i', audio_output_path, 
        '-c:v', 'copy', 
        '-c:a', 'aac', 
        '-shortest',  # Ensures no additional frames are added after the video ends
        final_output_path
    ])

    # Clean up temporary files
    os.remove(audio_output_path)
    os.remove(video_output_path)

    return final_output_path

@app.route('/')
def index():
    return render_template('fileupload.html')




# @app.route('/download', methods=['POST'])
# def download():
#     if 'video' not in request.files or 'wallpaper' not in request.files:
#         return jsonify({"error": "No file part"}), 400

#     video_file = request.files['video']
#     wallpaper_file = request.files['wallpaper']

#     if video_file.filename == '' or wallpaper_file.filename == '':
#         return jsonify({"error": "No selected file"}), 400

#     video_filename = secure_filename(video_file.filename)
#     input_video_path = os.path.join(UPLOAD_FOLDER, video_filename)
#     video_file.save(input_video_path)

#     wallpaper_filename = secure_filename(wallpaper_file.filename)
#     wallpaper_path = os.path.join(UPLOAD_FOLDER, wallpaper_filename)
#     wallpaper_file.save(wallpaper_path)

#     try:
#         # Process the video with the chosen wallpaper
#         processed_video_path = process_video(input_video_path, wallpaper_path)
#     except Exception as e:
#         return jsonify({"error": f"Error processing video: {e}"}), 500

#     # Clean up the original uploaded files
#     os.remove(input_video_path)
#     os.remove(wallpaper_path)

#     # Return the path to the processed video
#     return jsonify({
#         "message": "Video processed successfully",
#         "video_url": f"/uploads/{os.path.basename(processed_video_path)}"
#     })



@app.route('/download', methods=['POST'])
def download():
    if 'video' not in request.files:
        return jsonify({"error": "No video file part"}), 400

    video_file = request.files['video']
    video_filename = secure_filename(video_file.filename)
    input_video_path = os.path.join(UPLOAD_FOLDER, video_filename)
    video_file.save(input_video_path)

    wallpaper_path = None

    # Check for wallpaper input
    if 'wallpaper' in request.files and request.files['wallpaper'].filename != '':
        wallpaper_file = request.files['wallpaper']
        wallpaper_filename = secure_filename(wallpaper_file.filename)
        wallpaper_path = os.path.join(UPLOAD_FOLDER, wallpaper_filename)
        wallpaper_file.save(wallpaper_path)
    else:
        # Handle static wallpaper selection
        static_wallpaper = request.form.get('static_wallpaper')
        if static_wallpaper:
            wallpaper_path = os.path.join('static', static_wallpaper)

    if wallpaper_path is None:
        return jsonify({"error": "No wallpaper selected"}), 400

    try:
        # Process the video with the chosen wallpaper
        processed_video_path = process_video(input_video_path, wallpaper_path)
    except Exception as e:
        return jsonify({"error": f"Error processing video: {e}"}), 500

    # Clean up the original uploaded files
    os.remove(input_video_path)

    # Return the path to the processed video
    return jsonify({
        "message": "Video processed successfully",
        "video_url": f"/uploads/{os.path.basename(processed_video_path)}"
    })




@app.route('/convert', methods=['POST'])
def convert():
    if 'video' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    input_video_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_video_path)
    output_video_path = os.path.join(UPLOAD_FOLDER, filename.replace(".mp4", "_processed.mp4"))

    try:
        processed_video_path = process_video(input_video_path, output_video_path, background_type='black')
    except Exception as e:
        return jsonify({"error": f"Error processing video: {e}"}), 500

    os.remove(input_video_path)

    return jsonify({
        "message": "Video processed successfully",
        "video_url": f"/uploads/{os.path.basename(processed_video_path)}"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)








