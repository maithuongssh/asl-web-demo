from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import base64
import time
import threading
import os
from ultralytics import YOLO
import pyttsx3
import logging

app = Flask(__name__)

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thư mục upload ảnh
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load mô hình YOLO đã huấn luyện
model_path = "best.pt"
try:
    model = YOLO(model_path)
    logger.info(f"Model loaded successfully from {model_path}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# Danh sách lớp ký hiệu
class_names = [
    'a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y'
]

# Biến toàn cục
camera = None
is_camera_active = False
last_detection = None
last_detection_time = 0
detection_delay = 2  # Giảm từ 3 xuống 2 giây để phản hồi nhanh hơn
recognized_sequence = ""
confidence_threshold = 0.5  # Ngưỡng độ tin cậy tối thiểu
camera_lock = threading.Lock()

def get_camera():
    """Khởi tạo camera một cách an toàn"""
    global camera
    with camera_lock:
        if camera is None:
            try:
                # Thử các chỉ số camera khác nhau
                for i in range(5):
                    try:
                        camera = cv2.VideoCapture(i)
                        if camera.isOpened():
                            # Test read một frame
                            ret, frame = camera.read()
                            if ret and frame is not None:
                                logger.info(f"Camera initialized successfully on index {i}")
                                break
                            else:
                                camera.release()
                                camera = None
                    except Exception as e:
                        logger.warning(f"Failed to initialize camera on index {i}: {e}")
                        if camera:
                            camera.release()
                        camera = None
                        continue
                
                if camera is None or not camera.isOpened():
                    logger.error("Cannot open any camera")
                    return None
                
                # Cài đặt độ phân giải
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                camera.set(cv2.CAP_PROP_FPS, 30)
                
                # Kiểm tra lại camera hoạt động
                ret, test_frame = camera.read()
                if not ret or test_frame is None:
                    logger.error("Camera opened but cannot read frames")
                    camera.release()
                    camera = None
                    return None
                    
                logger.info("Camera initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing camera: {e}")
                if camera:
                    try:
                        camera.release()
                    except:
                        pass
                camera = None
                return None
        return camera

def release_camera():
    """Giải phóng camera một cách an toàn"""
    global camera, is_camera_active
    with camera_lock:
        is_camera_active = False
        if camera is not None:
            try:
                camera.release()
                logger.info("Camera released")
            except Exception as e:
                logger.error(f"Error releasing camera: {e}")
            finally:
                camera = None

def detect_sign_language(image):
    """Nhận diện ngôn ngữ ký hiệu từ ảnh"""
    if model is None:
        return None
    
    try:
        results = model(image, conf=confidence_threshold, verbose=False)
        detections = []
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    if cls < len(class_names) and conf >= confidence_threshold:
                        detections.append({
                            'class': class_names[cls],
                            'confidence': conf
                        })
        
        if detections:
            # Trả về detection có độ tin cậy cao nhất
            return max(detections, key=lambda x: x['confidence'])
        return None
    except Exception as e:
        logger.error(f"Error in detection: {e}")
        return None

def generate_frames():
    """Tạo stream video frames"""
    global is_camera_active, last_detection, last_detection_time, recognized_sequence
    
    camera = get_camera()
    if camera is None:
        logger.error("Cannot start camera stream")
        return
    
    is_camera_active = True
    logger.info("Starting video stream")
    
    try:
        frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Tính toán vị trí khung nhận diện (góc phải)
        box_size = 300
        box_x1 = frame_width - box_size - 20
        box_y1 = frame_height // 2 - box_size // 2
        box_x2 = box_x1 + box_size
        box_y2 = box_y1 + box_size

        frame_count = 0
        while is_camera_active:
            try:
                success, frame = camera.read()
                if not success or frame is None:
                    logger.error("Failed to read frame from camera")
                    time.sleep(0.1)
                    continue

                frame_count += 1
                
                # Lật frame theo chiều ngang để tạo hiệu ứng gương
                frame = cv2.flip(frame, 1)
                
                # Vẽ khung nhận diện
                cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (255, 0, 0), 2)
                cv2.putText(frame, "Detection Area", (box_x1, box_y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Lấy vùng quan tâm (ROI)
                roi = frame[box_y1:box_y2, box_x1:box_x2]

                # Thực hiện nhận diện với delay (chỉ mỗi 30 frames để tối ưu)
                if frame_count % 30 == 0:
                    current_time = time.time()
                    if current_time - last_detection_time >= detection_delay:
                        detection_result = detect_sign_language(roi)
                        if detection_result:
                            detected_char = detection_result['class']
                            confidence = detection_result['confidence']
                            
                            # Chỉ thêm vào sequence nếu khác ký tự trước đó
                            if detected_char != last_detection:
                                last_detection = detected_char
                                recognized_sequence += detected_char
                                logger.info(f"Detected: {detected_char} (confidence: {confidence:.2f})")
                                logger.info(f"Current sequence: {recognized_sequence}")
                            
                            last_detection_time = current_time

                # Hiển thị thông tin trên frame
                if last_detection:
                    cv2.putText(frame, f"Last: {last_detection}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.putText(frame, f"Sequence: {recognized_sequence}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # Encode frame thành JPEG với chất lượng tốt
                encode_param = [cv2.IMWRITE_JPEG_QUALITY, 80]
                ret, buffer = cv2.imencode('.jpg', frame, encode_param)
                if not ret:
                    logger.warning("Failed to encode frame")
                    continue
                    
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # Nhỏ delay để tránh CPU quá tải
                time.sleep(0.033)  # ~30 FPS
                       
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                time.sleep(0.1)
                continue
                   
    except Exception as e:
        logger.error(f"Error in generate_frames: {e}")
    finally:
        is_camera_active = False
        logger.info("Video stream ended")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Stream video feed"""
    try:
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logger.error(f"Error in video_feed: {e}")
        return Response("Error in video stream", status=500)

@app.route('/start_camera')
def start_camera():
    global is_camera_active
    try:
        if is_camera_active:
            return jsonify({"status": "already_active"})
            
        camera = get_camera()
        if camera is None:
            return jsonify({"error": "Cannot initialize camera. Please check if camera is connected and not being used by another application."}), 500
        
        # Đánh dấu camera đã sẵn sàng
        return jsonify({"status": "started", "message": "Camera initialized successfully"})
    except Exception as e:
        logger.error(f"Error starting camera: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/stop_camera')
def stop_camera():
    try:
        release_camera()
        return jsonify({"status": "stopped"})
    except Exception as e:
        logger.error(f"Error stopping camera: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/camera_status')
def camera_status():
    """Kiểm tra trạng thái camera"""
    return jsonify({
        "is_active": is_camera_active,
        "camera_available": camera is not None
    })

@app.route('/get_last_detection')
def get_last_detection():
    global last_detection
    return jsonify({"detection": last_detection if last_detection else ""})

@app.route('/reset_detection')
def reset_detection():
    global last_detection
    last_detection = None
    return jsonify({"status": "reset"})

@app.route('/upload_image', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Kiểm tra định dạng file
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        if not ('.' in file.filename and 
                file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({"error": "Invalid file format"}), 400

        # Lưu file
        filename = f"{int(time.time())}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Đọc và xử lý ảnh
        image = cv2.imread(filepath)
        if image is None:
            os.remove(filepath)  # Xóa file lỗi
            return jsonify({"error": "Cannot read image file"}), 400

        detection_result = detect_sign_language(image)

        if detection_result:
            return jsonify({
                "result": detection_result['class'],
                "confidence": float(detection_result['confidence']),
                "file_path": filepath
            })
        else:
            return jsonify({"error": "Không nhận diện được ký hiệu"})
            
    except Exception as e:
        logger.error(f"Error in upload_image: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/get_sequence')
def get_sequence():
    global recognized_sequence
    return jsonify({"sequence": recognized_sequence})

@app.route('/reset_sequence')
def reset_sequence():
    global recognized_sequence
    recognized_sequence = ""
    return jsonify({"status": "reset"})

@app.route('/clear_last_char')
def clear_last_char():
    """Xóa ký tự cuối cùng trong sequence"""
    global recognized_sequence
    if recognized_sequence:
        recognized_sequence = recognized_sequence[:-1]
    return jsonify({"status": "cleared", "sequence": recognized_sequence})

@app.route('/speak_sequence')
def speak_sequence():
    global recognized_sequence
    try:
        if recognized_sequence:
            # Chạy text-to-speech trong thread riêng để không block
            thread = threading.Thread(target=speak_text, args=(recognized_sequence,))
            thread.daemon = True
            thread.start()
            return jsonify({"status": "speaking", "text": recognized_sequence})
        return jsonify({"status": "empty"})
    except Exception as e:
        logger.error(f"Error in speak_sequence: {e}")
        return jsonify({"error": str(e)}), 500

def speak_text(text):
    """Text-to-speech function"""
    try:
        engine = pyttsx3.init()
        
        # Cài đặt thuộc tính giọng nói
        voices = engine.getProperty('voices')
        if voices:
            # Chọn giọng nữ nếu có
            for voice in voices:
                if 'female' in voice.name.lower() or 'woman' in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
        
        engine.setProperty('rate', 150)  # Tốc độ nói
        engine.setProperty('volume', 0.9)  # Âm lượng
        
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        logger.error(f"Error in text-to-speech: {e}")

@app.route('/set_confidence_threshold', methods=['POST'])
def set_confidence_threshold():
    """Thiết lập ngưỡng độ tin cậy"""
    global confidence_threshold
    try:
        data = request.get_json()
        threshold = float(data.get('threshold', 0.5))
        if 0.0 <= threshold <= 1.0:
            confidence_threshold = threshold
            return jsonify({"status": "updated", "threshold": confidence_threshold})
        else:
            return jsonify({"error": "Threshold must be between 0 and 1"}), 400
    except Exception as e:
        logger.error(f"Error setting confidence threshold: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_stats')
def get_stats():
    """Lấy thống kê ứng dụng"""
    return jsonify({
        "is_camera_active": is_camera_active,
        "sequence_length": len(recognized_sequence),
        "last_detection": last_detection,
        "confidence_threshold": confidence_threshold,
        "supported_classes": class_names
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
        release_camera()
    except Exception as e:
        logger.error(f"Application error: {e}")
        release_camera()