import cv2 as cv
import numpy as np
import time
from collections import deque
import math
import json

# =========================
# Config (외부 파일에서 로드)
# =========================
try:
    with open("config.json", "r") as f:
        CFG = json.load(f)
    print("✅ 설정(config.json)을 성공적으로 로드했습니다.")
except FileNotFoundError:
    print("❌ ERROR: config.json 파일을 찾을 수 없습니다. 프로그램을 종료합니다.")
    exit()

# =========================
# MediaPipe FaceMesh 준비
# =========================
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh

LMK = {
    "LEFT_EYE":  {"OUTER": 33, "INNER": 133, "UP": 159, "DOWN": 145},
    "RIGHT_EYE": {"OUTER": 263, "INNER": 362, "UP": 386, "DOWN": 374},
    "MOUTH": {"UP": 13, "DOWN": 14},
    "NOSE": 1,
    "L_IRIS": [468, 469, 470, 471, 472],
    "R_IRIS": [473, 474, 475, 476, 477]
}

# ===============
# 보조 유틸 함수
# ===============
def l2(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def safe_std(arr):
    if len(arr) < 2: return 0.0
    return float(np.std(arr))

def angle_deg(p1, p2):
    return math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]))

def normalize_scalar(x, eps=1e-6):
    return float(max(min(x, 1e3), -1e3))

# =========================
# 얼굴/ROI/지표 계산 함수
# =========================
def eye_open_ratio(landmarks):
    L = LMK["LEFT_EYE"]; R = LMK["RIGHT_EYE"]
    vL = l2(landmarks[L["UP"]], landmarks[L["DOWN"]])
    hL = l2(landmarks[L["OUTER"]], landmarks[L["INNER"]])
    earL = vL / (hL + 1e-6)
    vR = l2(landmarks[R["UP"]], landmarks[R["DOWN"]])
    hR = l2(landmarks[R["OUTER"]], landmarks[R["INNER"]])
    earR = vR / (hR + 1e-6)
    return (earL + earR) / 2.0

def iris_centers(landmarks):
    try:
        Li = np.array([landmarks[i] for i in LMK["L_IRIS"]])
        Ri = np.array([landmarks[i] for i in LMK["R_IRIS"]])
        if Li.size == 0 or Ri.size == 0:
            return None, None, None
        cL = Li.mean(axis=0)
        cR = Ri.mean(axis=0)
        center = ((cL[0]+cR[0])/2.0, (cL[1]+cR[1])/2.0)
        return (cL[0], cL[1]), (cR[0], cR[1]), center
    except (IndexError, ValueError):
        return None, None, None

def eyelid_distance(landmarks):
    L = LMK["LEFT_EYE"]; R = LMK["RIGHT_EYE"]
    dL = l2(landmarks[L["UP"]], landmarks[L["DOWN"]])
    dR = l2(landmarks[R["UP"]], landmarks[R["DOWN"]])
    return (dL + dR) / 2.0

def mouth_open_amount(landmarks):
    M = LMK["MOUTH"]
    L = LMK["LEFT_EYE"]; R = LMK["RIGHT_EYE"]
    face_w = l2(landmarks[L["OUTER"]], landmarks[R["OUTER"]]) + 1e-6
    d = l2(landmarks[M["UP"]], landmarks[M["DOWN"]]) / face_w
    return d

def head_pose_proxies(landmarks):
    L = LMK["LEFT_EYE"]; R = LMK["RIGHT_EYE"]
    nose = landmarks[LMK["NOSE"]]
    left_outer  = landmarks[L["OUTER"]]
    right_outer = landmarks[R["OUTER"]]
    roll = angle_deg(left_outer, right_outer)
    w = l2(left_outer, right_outer) + 1e-6
    yaw = (nose[0] - (left_outer[0]+right_outer[0])/2.0) / w
    eye_y = (left_outer[1] + right_outer[1]) / 2.0
    pitch = (nose[1] - eye_y) / w
    return normalize_scalar(roll), normalize_scalar(yaw), normalize_scalar(pitch)

def roi_boxes_from_face(frame_shape, face_bbox_norm):
    H, W = frame_shape[:2]
    x, y, w, h = face_bbox_norm
    x1 = int(max(0, x*W)); y1 = int(max(0, y*H))
    x2 = int(min(W, (x+w)*W)); y2 = int(min(H, (y+h)*H))
    face_box = (x1, y1, x2, y2)
    body_h = int(h*H*1.2)
    body_w = int(w*W*1.4)
    cx = (x1 + x2)//2
    bx1 = max(0, cx - body_w//2)
    bx2 = min(W, cx + body_w//2)
    by1 = min(H-1, y2 + int(0.1*h*H))
    by2 = min(H, by1 + body_h)
    body_box = (bx1, by1, bx2, by2)
    return face_box, body_box

def frame_diff_magnitude(prev_gray, gray, box):
    x1,y1,x2,y2 = box
    roi_prev = prev_gray[y1:y2, x1:x2]
    roi_now  = gray[y1:y2, x1:x2]
    if roi_prev.size == 0 or roi_now.size == 0: return 0.0
    diff = cv.absdiff(roi_prev, roi_now)
    return float(np.mean(diff))

# =========================
# 윈도/에포크 누적 버퍼
# =========================
class WindowStats:
    def __init__(self):
        self.reset()
    def reset(self):
        self.eye_open, self.iris_center, self.eyelid_dist, self.mouth_open = [], [], [], []
        self.roll, self.yaw, self.pitch = [], [], []
        self.eye_band_activity, self.face_diff, self.body_diff = [], [], []
        self.valid_frames = 0
    def add(self, sample):
        if not sample["valid"]: return
        self.valid_frames += 1
        self.eye_open.append(sample["eye_open"])
        self.iris_center.append(sample["iris_center"])
        self.eyelid_dist.append(sample["eyelid_dist"])
        self.mouth_open.append(sample["mouth_open"])
        self.roll.append(sample["roll"]); self.yaw.append(sample["yaw"]); self.pitch.append(sample["pitch"])
        self.eye_band_activity.append(abs(sample["eye_band_delta"]))
        self.face_diff.append(sample["face_diff"])
        self.body_diff.append(sample["body_diff"])
    def summarize(self):
        if self.valid_frames == 0: return {"valid": False}
        ic = np.array(self.iris_center)
        gaze_jitter = float(np.mean([safe_std(ic[:,0]), safe_std(ic[:,1])]))
        eye_band = float(np.mean(self.eye_band_activity)) if self.eye_band_activity else 0.0
        return {
            "valid": True, "eye_open_mean": float(np.mean(self.eye_open)),
            "eye_closed_frac": float(np.mean([1.0 if v < CFG["EYE_OPEN_THR"] else 0.0 for v in self.eye_open])),
            "gaze_jitter_std": gaze_jitter, "eyelid_band_mean": eye_band,
            "mouth_open_mean": float(np.mean(self.mouth_open)),
            "roll_std": safe_std(self.roll), "yaw_std": safe_std(self.yaw), "pitch_std": safe_std(self.pitch),
            "face_diff_mean": float(np.mean(self.face_diff)) if self.face_diff else 0.0,
            "body_diff_mean": float(np.mean(self.body_diff)) if self.body_diff else 0.0,
        }

class EpochStats:
    def __init__(self):
        self.short_summaries = deque()
        self.reset_counters()
    def add_short_summary(self, s):
        if s["valid"]:
            self.short_summaries.append(s)
            if len(self.short_summaries) > CFG["EPOCH_SEC"]: self.short_summaries.popleft()
    def add_frame_flags(self, face_diff, body_diff):
        if face_diff < CFG["FRAME_DIFF_THR_FACE"] and body_diff < CFG["FRAME_DIFF_THR_BODY"]: self.still_frames += 1
        if face_diff >= CFG["FRAME_DIFF_THR_FACE"]*2 or body_diff >= CFG["FRAME_DIFF_THR_BODY"]*2: self.burst_count += 1
        self.total_frames += 1
    def add_saccade_burst(self): self.saccade_bursts += 1
    def is_full(self): return len(self.short_summaries) >= CFG["EPOCH_SEC"]

    def reset_counters(self):
        self.saccade_bursts = 0
        self.burst_count = 0
        self.still_frames = 0
        self.total_frames = 0

    def compute_scores(self):
        if not self.is_full() or self.total_frames == 0: return 0.0, 0.0, {}
        shorts = list(self.short_summaries)
        eye_closed_frac_epoch = np.mean([s["eye_closed_frac"] for s in shorts])
        jitter_high_ratio = np.mean([1.0 if s["gaze_jitter_std"] > CFG["GAZE_JITTER_STD_THR"] else 0.0 for s in shorts])
        eyelid_band_high_ratio = np.mean([1.0 if s["eyelid_band_mean"] > CFG["EYE_BAND_ACTIVITY_THR"] else 0.0 for s in shorts])
        mouth_open_ratio = np.mean([1.0 if s["mouth_open_mean"] > CFG["MOUTH_OPEN_THR"] else 0.0 for s in shorts])
        mean_face = np.mean([s["face_diff_mean"] for s in shorts])
        mean_body = np.mean([s["body_diff_mean"] for s in shorts]) + 1e-6
        face_body_ratio = mean_face / mean_body
        head_std_sum = np.mean([s["roll_std"] + s["yaw_std"] + s["pitch_std"] for s in shorts])
        still_rate = self.still_frames / float(self.total_frames)
        breath_std = np.std([s["body_diff_mean"] for s in shorts])
        wR = CFG["WEIGHTS"]["REM"]; wN = CFG["WEIGHTS"]["N3"]
        rem_score, n3_score, details = 0.0, 0.0, {}
        if eye_closed_frac_epoch >= CFG["EYE_STABLE_FRAC_THR"]: n3_score += wN["eye_closed_stable"]; details["eye_closed_stable"] = True
        if still_rate >= CFG["STILL_FRAC_THR"]: n3_score += wN["still_high"]; details["still_high"] = True
        if eye_closed_frac_epoch >= 0.4 and jitter_high_ratio > CFG["GAZE_JITTER_EPOCH_RATIO_THR"] and eyelid_band_high_ratio > 0.2: rem_score += wR["eye_closed_and_micro"]; details["eye_closed_and_micro"] = True
        if self.saccade_bursts >= CFG["SACCADE_BURST_COUNT_THR"]: rem_score += wR["saccade_bursts"]; details["saccade_bursts"] = True
        if jitter_high_ratio > CFG["GAZE_JITTER_EPOCH_RATIO_THR"]: rem_score += wR["gaze_jitter"]; details["gaze_jitter"] = True
        if eyelid_band_high_ratio > CFG["EYE_BAND_EPOCH_RATIO_THR"]: rem_score += wR["eyelid_band"]; details["eyelid_band"] = True
        if still_rate >= 0.5 and (jitter_high_ratio > 0.4 or self.saccade_bursts > 10): rem_score += wR["still_but_micro"]; details["still_but_micro"] = True
        if face_body_ratio >= CFG["FACE_BODY_RATIO_THR"]: rem_score += wR["face_body_ratio"]; details["face_body_ratio"] = True
        if head_std_sum <= CFG["HEAD_STD_THR"]:
            n3_score += wN["head_very_still"]; details["head_very_still"] = True
            if CFG["HEAD_VS_EYE_REM_RULE"] and (jitter_high_ratio > 0.4 or self.saccade_bursts > 10): rem_score += wR["head_still_eye_active"]; details["head_still_eye_active"] = True
        if breath_std <= CFG["BREATH_RHYTHM_STD_THR"]: n3_score += wN["breath_regular"]; details["breath_regular"] = True
        if mouth_open_ratio >= CFG["MOUTH_OPEN_EPOCH_RATIO_THR"]: n3_score += wN["mouth_open"]; details["mouth_open"] = True
        return rem_score, n3_score, {
            "eye_closed_frac_epoch": eye_closed_frac_epoch, "jitter_high_ratio": jitter_high_ratio,
            "eyelid_band_high_ratio": eyelid_band_high_ratio, "face_body_ratio": face_body_ratio,
            "head_std_sum": head_std_sum, "still_rate": still_rate, "breath_std": breath_std,
            "mouth_open_ratio": mouth_open_ratio, "saccade_bursts": self.saccade_bursts,
            "burst_count": self.burst_count, "rem_details": details
        }

# =========================
# 메인 루프
# =========================
def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        return

    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing, mp_styles = mp.solutions.drawing_utils, mp.solutions.drawing_styles

    prev_gray, prev_eyelid, prev_gaze_center = None, None, None
    epoch, short = EpochStats(), WindowStats()

    in_deep = False
    deep_start_ts = None
    n3_miss_counter = 0

    last_short_ts = time.time()
    last_epoch_label = "UNK"

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv.flip(frame, 1)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        valid, eye_open_val, eyelid_dist_val, mouth_open_val = False, 0.0, 0.0, 0.0
        roll, yaw, pitch, eye_band_delta = 0.0, 0.0, 0.0, 0.0
        iris_center = (0.0, 0.0)

        out = face_mesh.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        face_box_norm = (0.35, 0.2, 0.3, 0.45)
        if out.multi_face_landmarks:
            lms = out.multi_face_landmarks[0].landmark
            pts = [(l.x, l.y) for l in lms]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            minx, maxx = max(min(xs), 0.0), min(max(xs), 1.0)
            miny, maxy = max(min(ys), 0.0), min(max(ys), 1.0)
            face_box_norm = (minx, miny, maxx - minx, maxy - miny)
            eye_open_val, eyelid_dist_val, mouth_open_val = eye_open_ratio(pts), eyelid_distance(pts), mouth_open_amount(pts)
            roll, yaw, pitch = head_pose_proxies(pts)

            _, _, gaze_center = iris_centers(pts)
            if gaze_center is not None:
                iris_center = (float(gaze_center[0]), float(gaze_center[1]))
                if prev_gaze_center is not None:
                    speed = l2(iris_center, prev_gaze_center)
                    if speed > CFG["SACCADE_SPEED_THR"]: epoch.add_saccade_burst()
                prev_gaze_center = iris_center

            if prev_eyelid is not None: eye_band_delta = eyelid_dist_val - prev_eyelid
            prev_eyelid = eyelid_dist_val
            valid = True

        face_box, body_box = roi_boxes_from_face(frame.shape, face_box_norm)
        face_diff, body_diff = 0.0, 0.0
        if prev_gray is not None:
            face_diff = frame_diff_magnitude(prev_gray, gray, face_box)
            body_diff = frame_diff_magnitude(prev_gray, gray, body_box)
            epoch.add_frame_flags(face_diff, body_diff)
        prev_gray = gray

        short.add({"valid": valid, "eye_open": eye_open_val, "iris_center": iris_center, "eyelid_dist": eyelid_dist_val, "eye_band_delta": eye_band_delta, "mouth_open": mouth_open_val, "roll": roll, "yaw": yaw, "pitch": pitch, "face_diff": face_diff, "body_diff": body_diff})

        now = time.time()
        if now - last_short_ts >= CFG["SHORT_WIN_SEC"]:
            epoch.add_short_summary(short.summarize())
            short.reset()
            last_short_ts = now

            if epoch.is_full():
                rem_score, n3_score, info = epoch.compute_scores()
                if rem_score >= CFG["REM_THRESHOLD"] and rem_score > n3_score: label = "REM_LIKE"
                elif n3_score >= CFG["N3_THRESHOLD"] and n3_score > rem_score: label = "N3_LIKE"
                else: label = "UNK"

                if label == "N3_LIKE":
                    n3_miss_counter = 0
                    if not in_deep:
                        if deep_start_ts is None: deep_start_ts = now
                        elif now - deep_start_ts >= CFG["DEEP_MIN_SECONDS"]:
                            in_deep = True
                else:
                    n3_miss_counter += 1
                    if n3_miss_counter > CFG["N3_MISS_TOLERANCE"]:
                        deep_start_ts = None
                        in_deep = False

                last_epoch_label = label
                print(f"[30s] REM={rem_score:.1f}  N3={n3_score:.1f}  → {label} (Deep: {in_deep})")
                epoch.reset_counters()

        # 화면 오버레이
        cv.rectangle(frame, (face_box[0], face_box[1]), (face_box[2], face_box[3]), (0,255,0), 2)
        cv.rectangle(frame, (body_box[0], body_box[1]), (body_box[2], body_box[3]), (255,0,0), 2)
        y0 = 24
        def put(txt): nonlocal y0; cv.putText(frame, txt, (10, y0), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv.LINE_AA); y0 += 22
        put(f"Label: {last_epoch_label}   Deep: {in_deep}")
        put(f"EyeOpen: {eye_open_val:.3f}   MouthOpen: {mouth_open_val:.3f}")
        put(f"Roll/Yaw/Pitch: {roll:.1f}/{yaw:.3f}/{pitch:.3f}")
        put(f"FaceDiff: {face_diff:.1f}  BodyDiff: {body_diff:.1f}")

        # NEW: 얼굴 외곽선, 눈, 입 등 랜드마크 그리기
        if out.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=out.multi_face_landmarks[0],
                connections=mp_face_mesh.FACEMESH_CONTOURS, # FACEMESH_CONTOURS: 외곽선만 그림
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()
            )
        # --- END OF NEW SECTION ---

        cv.imshow("SomnoTrack REM/N3 Analyzer", frame)
        if cv.waitKey(1) == 27: break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()