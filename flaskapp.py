from flask import Flask, Response, request, jsonify
from flask_cors import CORS

import cv2

from yolo_video import video_detection

app = Flask(__name__)
CORS(app)

app.config['SECRET_KEY'] = 'idpc3'

def generate_frames(path_x = ''):

    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref,buffer = cv2.imencode('.jpg', detection_)

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
@app.route('/webcam')
def webcam():
    return Response(generate_frames(path_x=0), mimetype='multipart/X-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0')