import cv2
import dlib
import numpy as np
import pandas as pd
from imutils import face_utils

# Initialize dlib's face detector and facial landmark predictor
predictor_path = r"C:\shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Video paths
video_paths = [
    r"D:\Data\Crisis\Cropped files\Group 1\Grp 1 Crop Videos to mp4\4107_Charles Schwab Corporation_Charles Schwab_071808.mp4",
    r"D:\Data\Crisis\Cropped files\Group 1\Grp 1 Crop Videos to mp4\4137_Citizens Republic Bancorp_William Hartman_072108.mp4"
]

# Variables for storing results
frame_number = 0
results = []
min_face_area = 1000  # Minimum face area to consider (to exclude very small faces)
face_id_counter = 0   # Counter to assign unique IDs to faces
face_trackers = {}    # Dictionary to track face locations and assign IDs

# Function to calculate midpoint
def midpoint(p1, p2):
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

# Function to calculate fWHR (facial width-to-height ratio)
def calculate_fWHR(shape):
    P2 = shape[1]   # Point 2
    P16 = shape[15]  # Point 16
    face_width = np.linalg.norm(P2 - P16)

    P22 = shape[21]  # Point 22
    P23 = shape[22]  # Point 23
    midpoint_P22_P23 = midpoint(P22, P23)
    P52 = shape[51]  # Point 52 (between the nose and lips)
    face_height = np.linalg.norm(np.array(midpoint_P22_P23) - P52)

    fWHR = face_width / face_height if face_height > 0 else 0
    return fWHR, face_width, face_height

# Function to calculate TW (trustworthiness index without fWHR)
def calculate_TW(eyebrow_angle, chin_angle, philtrum_length):
    if len(eyebrows) > 1 and len(chin_angles) > 1 and len(philtrums) > 1:
        eyebrow_std = (eyebrow_angle - np.mean(eyebrows)) / np.std(eyebrows)
        chin_angle_std = (chin_angle - np.mean(chin_angles)) / np.std(chin_angles) if np.std(chin_angles) != 0 else 0
        philtrum_std = (philtrum_length - np.mean(philtrums)) / np.std(philtrums)
        TW = (eyebrow_std * -1 + chin_angle_std + philtrum_std * -1) / 3
    else:
        TW = np.nan  # 데이터가 충분하지 않으면 TW는 NaN
    return TW

# Pre-calculated mean and std values for normalization (Example values, should be updated with actual data)
eyebrows = []
chin_angles = []
philtrums = []

# Track the faces and assign unique IDs
def assign_face_id(rect):
    global face_id_counter
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    face_area = w * h

    # Only consider large enough faces
    if face_area < min_face_area:
        return None

    # Calculate center of the face
    center = (x + w // 2, y + h // 2)

    # Try to match with existing faces using center proximity
    for face_id, (prev_center, prev_rect) in face_trackers.items():
        (px, py) = prev_center
        if abs(px - center[0]) < 50 and abs(py - center[1]) < 50:
            face_trackers[face_id] = (center, rect)  # Update tracking
            return face_id

    # If no match is found, assign a new face ID
    face_id_counter += 1
    face_trackers[face_id_counter] = (center, rect)
    return face_id_counter

# Process each video file
for video_path in video_paths:
    cap = cv2.VideoCapture(video_path)
    video_name = video_path.split("\\")[-1].split(".")[0]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"End of video or cannot read the file: {video_name}")
            break

        frame_number += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Detect faces in the grayscale frame with upsampling
        rects = detector(gray, 1)

        # Skip frames with no faces detected
        if len(rects) == 0:
            print(f"Skipping frame {frame_number}: No faces detected.")
            continue

        # Process each detected face
        for rect in rects:
            face_id = assign_face_id(rect)

            if face_id is None:
                continue  # Skip small faces or unmatched

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Calculate fWHR and face size
            fwh_ratio, face_width, face_height = calculate_fWHR(shape)

            # Calculate facial features for TW
            left_eyebrow_angle = np.arctan2(shape[22][1] - shape[21][1], shape[21][0] - shape[22][0]) * 180 / np.pi
            right_eyebrow_angle = np.arctan2(shape[24][1] - shape[23][1], shape[23][0] - shape[24][0]) * 180 / np.pi
            avg_eyebrow_angle = (left_eyebrow_angle + right_eyebrow_angle) / 2

            chin_angle = np.arctan2(shape[9][1] - shape[8][1], shape[8][0] - shape[9][0]) * 180 / np.pi
            philtrum_length = np.linalg.norm(shape[33] - shape[51])

            # Add to lists for normalization
            eyebrows.append(avg_eyebrow_angle)
            chin_angles.append(chin_angle)
            philtrums.append(philtrum_length)

            # Calculate TW (without fWHR)
            tw = calculate_TW(avg_eyebrow_angle, chin_angle, philtrum_length)

            # Store the results for each face in the current frame
            results.append({
                'Frame_Number': frame_number,
                'Face_ID': face_id,  # Assign the detected face's unique ID
                'fWHR': fwh_ratio,
                'TW': tw
            })

    cap.release()
    cv2.destroyAllWindows()

# DataFrame 생성
df = pd.DataFrame(results)

# 요약 통계 계산
summary_stats = df.groupby('Face_ID').agg({
    'fWHR': ['mean', 'std'],
    'TW': ['mean', 'std']
}).reset_index()

# MultiIndex를 평탄화하여 엑셀에 저장 가능하게 함
summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]

# 엑셀 파일로 저장
output_path = r"D:\Data\Crisis\Cropped files\fwhr and tw"
output_file = f"{output_path}\{video_name}_fwhr_tw_analysis_with_auto_face_id.xlsx"
with pd.ExcelWriter(output_file) as writer:
    df.to_excel(writer, sheet_name="Frame_Data", index=False)
    summary_stats.to_excel(writer, sheet_name="Summary_Statistics", index=False)

print(f"Results and summary saved successfully to {output_file}")
