# import cv2
# import dlib
# from imutils import face_utils
# import numpy as np
#
# # 加载人脸检测器和人脸关键点预测器
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # 确保文件路径正确
#
# # 定义嘴部关键点的索引范围
# (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
#
# # 初始化视频捕获（替换为你的视频文件路径或使用摄像头）
# cap = cv2.VideoCapture('26287540966-1-192.mp4')  # 如果要使用摄像头，替换为 0
#
# # 设置嘴部纵横比的阈值
# MAR_THRESH = 0.6
#
# def mouth_aspect_ratio(mouth):
#     # 计算嘴部纵横比（MAR）
#     A = np.linalg.norm(mouth[13] - mouth[19])  # 51, 59
#     B = np.linalg.norm(mouth[14] - mouth[18])  # 52, 58
#     C = np.linalg.norm(mouth[15] - mouth[17])  # 53, 57
#     D = np.linalg.norm(mouth[12] - mouth[16])  # 50, 54
#     mar = (A + B + C) / (3.0 * D)
#     return mar
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # 调整帧大小以加快处理速度（可选）
#     # frame = imutils.resize(frame, width=640)
#
#     # 转为灰度图像
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # 检测人脸
#     rects = detector(gray, 0)
#
#     for rect in rects:
#         # 获取人脸关键点
#         shape = predictor(gray, rect)
#         shape = face_utils.shape_to_np(shape)
#
#         # 提取嘴部坐标
#         mouth = shape[mStart:mEnd]
#
#         # 计算嘴部纵横比
#         mar = mouth_aspect_ratio(mouth)
#
#         # 绘制嘴部轮廓
#         cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1)
#
#         # 判断是否在说话
#         if mar > MAR_THRESH:
#             cv2.putText(frame, "Speaking", (rect.left(), rect.top() - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#         else:
#             cv2.putText(frame, "Not Speaking", (rect.left(), rect.top() - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#
#     # 显示结果
#     cv2.imshow("Frame", frame)
#     key = cv2.waitKey(1) & 0xFF
#
#     # 按'q'键退出
#     if key == ord("q"):
#         break
#
# # 释放资源
# cap.release()
# cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture('26167611388-1-192.mp4')

# 定义嘴部关键点索引
mouth_indices = [61, 291, 78, 308, 14, 13, 82, 312, 87, 317, 95, 324,
                 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270,
                 269, 267, 0, 37, 39, 40, 185, 95, 88, 178, 87, 14]

# 设置嘴部纵横比的阈值
MAR_THRESH = 0.6

def calculate_mar(mouth):
    # 将关键点坐标转换为NumPy数组
    mouth = np.array([(point[0], point[1]) for point in mouth])

    # 选取对应的关键点计算距离
    A = np.linalg.norm(mouth[13] - mouth[19])
    B = np.linalg.norm(mouth[14] - mouth[18])
    C = np.linalg.norm(mouth[15] - mouth[17])
    D = np.linalg.norm(mouth[12] - mouth[16])

    mar = (A + B + C) / (3.0 * D)
    return mar

# 初始化用于时间序列分析的变量
speaking_frames = 0
SPEAKING_CONSEC_FRAMES = 5  # 连续超过阈值的帧数

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 转为RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 提取嘴部关键点
            h, w, _ = frame.shape
            mouth = [(int(face_landmarks.landmark[i].x * w),
                      int(face_landmarks.landmark[i].y * h)) for i in mouth_indices]

            # 计算MAR
            mar = calculate_mar(mouth)

            # 绘制嘴部轮廓
            cv2.polylines(frame, [np.array(mouth)], True, (0, 255, 0), 1)

            # 判定说话状态
            if mar > MAR_THRESH:
                speaking_frames += 1
                if speaking_frames >= SPEAKING_CONSEC_FRAMES:
                    status = "Speaking"
                    cv2.putText(frame, status, (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                speaking_frames = 0
                status = "Not Speaking"
                cv2.putText(frame, status, (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        # 如果未检测到人脸，重置状态
        speaking_frames = 0
        status = "No Face Detected"
        cv2.putText(frame, status, (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 显示结果
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
