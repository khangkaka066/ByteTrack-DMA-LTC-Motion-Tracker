# Report: Dynamic Weight Network for Motion–Appearance Fusion in Multi-Object Tracking

## 1. Mục tiêu

Mục tiêu là xây dựng một module **Dynamic Weight Network** cho bài toán Multi-Object Tracking, nhằm tự động điều chỉnh mức độ tin tưởng giữa hai nguồn thông tin chính:

* **Motion cue**: thông tin chuyển động từ Kalman Filter.
* **Appearance cue**: thông tin nhận dạng ngoại hình từ ReID model.

Thay vì dùng trọng số cố định:

```text
Cost = 0.5 * Motion + 0.5 * ReID
```

hệ thống sẽ học trọng số động:

```text
Cost = w_motion * Cost_motion + w_reid * Cost_reid
```

trong đó:

```text
w_motion + w_reid = 1
```

và trọng số thay đổi theo từng track, từng detection, từng frame.

---

## 2. Vấn đề cần giải quyết

Trong MOT, Kalman Filter và ReID đều có điểm mạnh/yếu riêng.

Kalman Filter đáng tin khi:

* object di chuyển mượt
* camera ổn định
* không bị occlusion
* thời gian mất track ngắn

Kalman Filter kém tin cậy khi:

* camera motion mạnh
* object đổi hướng đột ngột
* bị che khuất lâu
* detector miss nhiều frame

ReID đáng tin khi:

* appearance rõ
* object bị mất track rồi xuất hiện lại
* motion prediction không còn chính xác

ReID kém tin cậy khi:

* nhiều người mặc giống nhau
* bbox nhỏ/mờ
* motion blur
* góc nhìn thay đổi mạnh
* embedding không ổn định

Vì vậy, cần một module học được:

```text
Khi nào nên tin motion nhiều hơn?
Khi nào nên tin appearance nhiều hơn?
```

---

## 3. Kiến trúc tổng thể

Pipeline đề xuất:

```text
Video frames
   ↓
Object Detector
   ↓
Existing Tracks
   ↓
Kalman Prediction
   ↓
ReID Feature Extraction
   ↓
Motion Cost + Appearance Cost
   ↓
Dynamic Weight Network
   ↓
Adaptive Fusion Cost
   ↓
Hungarian Matching
   ↓
Track Update
```

Module mới nằm ở bước association.

---

## 4. Input của hệ thống

### 4.1. Input tổng thể

Input chính của tracker là video sequence:

```text
Frame_1, Frame_2, ..., Frame_T
```

Mỗi frame được đưa qua detector để lấy:

```text
bbox, detection confidence, class
```

Với mỗi detection:

```text
d_j = [x1, y1, x2, y2, score]
```

---

## 5. Input của Dynamic Weight Network

Dynamic Weight Network nhận feature vector mô tả độ tin cậy của motion và appearance. Feature vector này được cài đặt cụ thể trong `yolox/DMA/features.py`, hàm `extract_pair_features()`, với `FEAT_DIM = 15`.

Với mỗi cặp:

```text
track_i, detection_j
```

ta tạo input feature `x_ij ∈ R^15`:

```text
x_ij = [
    motion features,        # idx 0-6
    appearance features,    # idx 7-8
    detection features,     # idx 9-11
    track history features, # idx 12-14
]
```

### 5.1. Motion features (idx 0–6)

Các feature liên quan đến Kalman Filter, tính từ `track.mean` / `track.covariance` (KalmanFilter) so với `detection.tlbr`:

```text
0  motion_iou         = IoU(predicted_bbox_i, detection_bbox_j)
1  motion_cost        = 1 - motion_iou
2  mahalanobis_norm   = clip(gating_distance(maha) / chi2inv95[4], 0, 1)
3  cov_trace_log      = log1p(trace(covariance[:4,:4]))       # tổng bất định vị trí
4  cov_mean_log       = log1p(mean(diag(covariance[:4,:4])))  # bất định trung bình
5  vel_magnitude      = tanh(sqrt(vx^2 + vy^2) / 10.0)         # tốc độ chuẩn hoá
6  time_since_update  = clip((current_frame - track.frame_id) / 300, 0, 1)
```

Ý nghĩa:

* `cov_trace_log` / `cov_mean_log` lớn → motion không chắc chắn (uncertainty-aware)
* `time_since_update` lớn → track đã mất lâu, Kalman kém tin cậy
* `motion_iou` thấp → prediction không khớp detection

---

### 5.2. Appearance features (idx 7–8)

Các feature liên quan đến ReID, tính từ `track.smooth_feat` (embedding trung bình mượt của track) và `detection.curr_feat`:

```text
7  cosine_dist    = clip(cosine_distance(track.smooth_feat, detection.curr_feat), 0, 1)
8  feat_variance  = mean(var(stack(track.features), axis=0))   # ≥ 2 embedding lịch sử
```

Nếu track hoặc detection không có embedding hợp lệ, dùng giá trị trung tính: `cosine_dist = 0.5`, `feat_variance = 0.0` (đánh dấu bởi `has_appearance = 0`, idx 14).

Ý nghĩa:

* `cosine_dist` thấp → hai object có appearance giống nhau
* `feat_variance` cao → embedding không ổn định qua thời gian → ReID kém tin cậy

---

### 5.3. Detection features (idx 9–11)

Các feature từ detector, tính từ `detection.score` và `detection.tlwh`:

```text
9   det_score      = detection.score
10  bbox_area_log  = log1p(w * h) / 15.0
11  bbox_aspect    = clip(w / h, 0.1, 10.0)
```

Ý nghĩa:

* `det_score` thấp → detection có thể sai
* `bbox_area_log` nhỏ → bbox nhỏ, ReID feature thường kém
* `bbox_aspect` bất thường → bbox bị méo, appearance không đáng tin

---

### 5.4. Track history features (idx 12–14)

Các feature từ lịch sử track:

```text
12  track_age_norm     = clip((current_frame - track.start_frame) / 300, 0, 1)
13  tracklet_len_norm  = clip(track.tracklet_len / 30, 0, 1)
14  has_appearance     = 1 nếu cả track và detection có embedding hợp lệ, ngược lại 0
```

Ý nghĩa:

* `track_age_norm` cao, `tracklet_len_norm` cao → track lâu và ổn định, motion có thể đáng tin hơn
* track mới tạo (age/len thấp) → chưa đủ lịch sử motion
* `has_appearance = 0` → không có tín hiệu ReID, network phải dựa chủ yếu vào motion

---

### 5.5. Cài đặt batch

`extract_batch_features(tracks, detections, kf, current_frame_id)` gọi `extract_pair_features()` cho toàn bộ cặp `(track_i, detection_j)` và trả về tensor `(n_tracks, n_detections, 15)`, dùng trực tiếp làm input batch cho MLP trước bước Hungarian matching.

---

### 5.6. Vì sao chọn các feature và công thức/hằng số này (rationale & limitations)

Nguyên tắc chung: ở bước association, model không chỉ cần biết "IoU cao hay thấp", "cosine gần hay xa" mà cần biết **cue đó có đáng tin ở tình huống này không** (reliability-aware, mục 18). Vì vậy mỗi nhóm feature vừa mang tín hiệu match, vừa mang tín hiệu độ tin cậy của chính cue đó.

**Motion (idx 0–6)**

* `motion_iou` / `motion_cost = 1 - iou`: tín hiệu match trực tiếp. Giữ cả hai tuy phụ thuộc tuyến tính nhau — cố ý dư thừa để MLP không phải tự học phép trừ, đổi lại chỉ tốn 1 chiều input.
* `mahalanobis_norm`: IoU không phân biệt được "lệch nhỏ nhưng track rất chắc chắn" với "lệch nhỏ nhưng track đang bất định". Mahalanobis (`kf.gating_distance`, `kalman_filter.py:262-268`) tự động chia theo covariance nên là bản IoU đã hiệu chỉnh theo độ bất định.
  * Chia cho `chi2inv95[4] = 9.4877`: ngưỡng gating chuẩn 95% với 4 bậc tự do (`kalman_filter.py:11-20`, lấy từ bảng chi-square của MATLAB/Octave `chi2inv`), đúng ngưỡng gating gốc của DeepSORT/ByteTrack — dùng lại để nhất quán về thang đo với bước gating khác trong pipeline, thay vì bịa một hằng số tuỳ ý.
* `cov_trace_log` / `cov_mean_log`: uncertainty-aware feature — track mất track lâu/occlusion khiến `covariance` (8×8 của Kalman) phình to do nhiễu cộng dồn qua các bước `predict()` không được `update()` điều chỉnh (`kalman_filter.py:107-117`). Dùng cả trace (tổng năng lượng bất định, nhạy với outlier ở 1 chiều) và mean diagonal (giá trị trung bình, ổn định hơn) vì chúng bổ sung cho nhau.
  * `log1p`: covariance có thể tăng gần cấp số nhân theo số frame mất track; nếu để tuyến tính, vài track "mất lâu" sẽ áp đảo gradient của MLP (không có input normalization riêng). `log1p` nén miền giá trị lớn, giữ thứ tự nhưng ổn định hơn khi train.
* `vel_magnitude = tanh(sqrt(vx²+vy²)/10.0)`: tốc độ thô không có upper bound; `tanh` nén về `[0,1)` để 1 track di chuyển bất thường nhanh không làm lệch gradient toàn cục.
  * Hằng số `10.0` (`_VEL_SCALE`): **không có công thức toán học chứng minh**, chỉ là giả định heuristic về "tốc độ điển hình" (px/frame) sao cho `tanh(v/10)` bão hoà ở tốc độ bất thường và vẫn nhạy ở tốc độ bình thường. Đây là hằng số yếu nhất trong toàn bộ thiết kế — nên tune lại theo dataset (người đi bộ ở MOT17 chậm hơn nhiều so với cầu thủ chạy ở SoccerNet/football).
* `time_since_update`: tín hiệu độ tin cậy quan trọng nhất của motion (track mất lâu → Kalman kém tin).
  * Hằng số `_MAX_AGE = 300`: cố ý chọn lớn hơn hẳn `buffer_size` thực tế (mặc định `track_buffer=30` frame ở 30fps, `byte_tracker.py:236`: `buffer_size = frame_rate/30 * track_buffer`) để feature không bão hoà về 1.0 ngay khi track vừa lost — cho phép phân biệt mượt "mất 5 frame" vs "mất 25 frame" trong toàn dải track còn sống. Hạn chế: đây là hardcode độc lập với config `track_buffer`, có thể lệch pha nếu ai đó đổi `track_buffer` ở nơi khác.

**Appearance (idx 7–8)**

* `cosine_dist`: so `track.smooth_feat` (EMA của embedding: `smooth_feat = alpha*smooth_feat + (1-alpha)*feat`, `alpha=0.9` mặc định, `byte_tracker.py:56`) với `detection.curr_feat`, không dùng embedding thô 1 frame vì EMA đã lọc bớt nhiễu tức thời (motion blur, occlusion 1 phần) — đúng tinh thần "track ổn định → appearance đáng tin hơn". `alpha=0.9` kế thừa từ code gốc ByteTrack/DeepSORT, không phải do `features.py` định nghĩa.
* `feat_variance`: đo độ ổn định embedding qua lịch sử (`track.features`, deque). Variance cao → nhiều khả năng gây nhầm ID (mục 5.2).
* Giá trị trung tính khi thiếu appearance (`cosine_dist=0.5`, `feat_variance=0.0`): chọn giữa khoảng thay vì 0/1 để không thiên vị "match"/"không match" khi hoàn toàn không có embedding — đây là lý do bắt buộc phải có thêm `has_appearance` (idx 14) để model biết phân biệt "giá trị thật" với "placeholder".

**Detection (idx 9–11)**

* `det_score`: score thấp → cả motion lẫn appearance tính từ detection đó đều kém tin cậy (insight cốt lõi ByteTrack gốc).
* `bbox_area_log = log1p(w*h)/15.0`: `log1p` để nén chênh lệch diện tích rất lớn giữa object nhỏ/lớn (cùng lý do với `cov_trace_log`). Hằng số `15.0` ước lượng từ `log1p(1920*1080) ≈ 14.54` (comment gốc "approx normalised for HD video") — chuẩn hoá gần về 1 khi bbox chiếm gần hết khung hình Full-HD. Hạn chế: gắn với 1 độ phân giải cụ thể, cần đổi hằng số nếu train trên video 4K hoặc SD.
* `bbox_aspect = clip(w/h, 0.1, 10)`: bbox méo do occlusion/detection lỗi lệch khỏi aspect ratio đặc trưng của người (~0.3–0.5) → appearance kém tin cậy. Clip chỉ là guard rail số học (tránh chia cho số rất nhỏ), không mang ý nghĩa thống kê như `chi2inv95`.

**Track history (idx 12–14)**

* `track_age_norm` vs `tracklet_len_norm`: đo hai thứ khác nhau — `track_age` tính từ `start_frame` (kể cả các đoạn bị lost), còn `tracklet_len` là số lần match liên tiếp không đứt quãng (reset về 0 mỗi khi track mất rồi được nối lại, `byte_tracker.py` phần `re_activate`). Một track "già nhưng vừa nối lại" (age lớn, len nhỏ) là tín hiệu khác hẳn track "già và liên tục" — cần cả hai để phân biệt.
  * `_MAX_LEN = 30` khớp trực tiếp với giá trị mặc định `--track_buffer 30` dùng trong toàn bộ `tools/track_*.py` — hằng số này có căn cứ rõ ràng từ config thực tế.
* `has_appearance`: cờ bắt buộc để tách "không có tín hiệu ReID" khỏi "có tín hiệu trung tính" (giải thích ở trên).

**Bảng tổng hợp mức độ có căn cứ của từng hằng số:**

| Hằng số | Căn cứ | Độ chắc chắn |
|---|---|---|
| `chi2inv95[4] = 9.4877` | Bảng thống kê chuẩn (chi-square 95%, 4 DOF), dùng lại từ Kalman gating gốc | Chắc chắn |
| `_MAX_LEN = 30` | Khớp `track_buffer` mặc định | Chắc chắn |
| `alpha = 0.9` (smooth_feat) | Kế thừa từ ByteTrack/DeepSORT gốc | Chắc chắn (ngoài phạm vi `features.py`) |
| `_MAX_AGE = 300` | Chọn ≈ 10× buffer_size để tránh bão hoà sớm | Hợp lý nhưng hardcode, chưa liên kết động với config `track_buffer` |
| `_VEL_SCALE = 10.0` | Heuristic "tốc độ điển hình" theo pixel/frame | Yếu nhất — cần tune theo dataset (MOT17 vs SoccerNet có tốc độ chuyển động khác hẳn nhau) |
| `bbox_area_log / 15.0` | Ước lượng cho Full-HD (1920×1080) | Gắn với 1 độ phân giải cụ thể — cần đổi nếu train ở resolution khác |

Bảng này nên được dùng làm cơ sở tham chiếu khi viết phần **Rủi ro** (mục 17) và thiết kế **Ablation** (mục 14) — đặc biệt các hằng số "yếu" (`_VEL_SCALE`, `_MAX_AGE`, `bbox_area_log/15`) là ứng viên tốt cho thí nghiệm sensitivity/ablation trước khi công bố kết quả.

---

## 6. Output của Dynamic Weight Network

Network output 2 logits:

```text
z_motion, z_reid
```

Sau đó dùng softmax:

```text
[w_motion, w_reid] = softmax([z_motion, z_reid])
```

Output cuối cùng:

```text
w_motion ∈ [0, 1]
w_reid ∈ [0, 1]
w_motion + w_reid = 1
```

Sau đó tính fusion cost:

```text
Cost_final = w_motion * Cost_motion + w_reid * Cost_reid
```

Ví dụ:

Trường hợp object di chuyển ổn định:

```text
w_motion = 0.8
w_reid = 0.2
```

Trường hợp object bị occlusion hoặc mất track lâu:

```text
w_motion = 0.3
w_reid = 0.7
```

---

## 7. Model đề xuất

Dynamic Weight Network có thể là MLP nhỏ:

```text
Input feature vector
   ↓
Linear
   ↓
ReLU
   ↓
Linear
   ↓
ReLU
   ↓
Linear
   ↓
Softmax
   ↓
[w_motion, w_reid]
```

Ví dụ kiến trúc (khớp `FEAT_DIM = 15` trong `features.py`):

```text
Input dim: 15
Hidden dim: 64
Hidden dim: 32
Output dim: 2
Activation: ReLU
Output activation: Softmax
```

Model này nhẹ, có thể chạy online trong tracker.

---

## 8. Dataset cần để train

Dataset cần có:

### 8.1. Video sequence

Mỗi dataset cần nhiều video có object chuyển động liên tục:

```text
Frame sequence
```

### 8.2. Bounding box annotation

Mỗi object cần bbox theo từng frame:

```text
frame_id, object_id, x, y, w, h
```

### 8.3. Identity annotation

Cần ID nhất quán qua thời gian:

```text
object_id
```

Đây là phần quan trọng nhất để train association.

### 8.4. Detection results

Có thể dùng:

* detection ground truth
* detection từ YOLO/Faster R-CNN
* public detections của MOT benchmark

Nên train/evaluate với detection thực tế để giống real-world hơn.

### 8.5. ReID embeddings

Cần crop object từ bbox rồi đưa qua ReID model để lấy embedding:

```text
embedding_dim = 128 / 256 / 512 / 2048
```

Có thể dùng ReID pretrained model, ví dụ:

```text
FastReID
OSNet
BoT-SORT ReID model
StrongSORT ReID model
```

---

## 9. Dataset phù hợp

### 9.1. MOT17

Phù hợp để làm baseline đầu tiên.

Ưu điểm:

* phổ biến
* nhiều tracker so sánh
* có ID annotation
* dễ báo cáo MOTA, IDF1, HOTA

### 9.2. MOT20

Khó hơn MOT17.

Phù hợp để kiểm tra trong cảnh đông người, occlusion nặng.

### 9.3. DanceTrack

Rất phù hợp cho nghiên cứu association.

Đặc điểm:

* chuyển động phức tạp
* appearance tương tự nhau
* motion khó dự đoán
* dễ gây ID switch

Nếu module dynamic weight tốt, nên test trên DanceTrack.

### 9.4. SoccerNet Tracking

Phù hợp nếu muốn ứng dụng vào bóng đá.

Đặc điểm:

* camera motion mạnh
* cầu thủ mặc đồng phục giống nhau
* occlusion nhiều
* ReID khó
* motion cũng khó

Đây là dataset tốt nếu muốn nhấn mạnh ứng dụng thể thao.

---

## 10. Cách tạo dữ liệu train cho Dynamic Weight Network

Từ ground truth, tạo các cặp:

```text
(track_i, detection_j)
```

Label:

```text
y_ij = 1 nếu track_i và detection_j cùng ID
y_ij = 0 nếu khác ID
```

Với mỗi cặp, lưu:

```text
features_ij
cost_motion_ij
cost_reid_ij
label_ij
```

Ví dụ một sample train:

```text
[
  IoU,
  Mahalanobis distance,
  Kalman covariance,
  time_since_update,
  detection confidence,
  bbox area,
  cosine distance,
  track age,
  velocity magnitude
]
→ label: same ID / different ID
```

---

## 11. Loss function

Có thể dùng 3 hướng.

### 11.1. Binary Classification Loss

Sau khi tính final cost, biến thành probability matching:

```text
p_match = sigmoid(-Cost_final)
```

Dùng Binary Cross Entropy:

```text
Loss = BCE(p_match, label)
```

Mục tiêu:

```text
same ID → cost thấp
different ID → cost cao
```

---

### 11.2. Ranking Loss

Với một track, detection đúng phải có cost thấp hơn detection sai:

```text
Cost_positive + margin < Cost_negative
```

Loss:

```text
Loss = max(0, margin + Cost_positive - Cost_negative)
```

Hướng này rất phù hợp với association.

---

### 11.3. Tracking-level Loss

Train trực tiếp theo kết quả tracking như IDF1/HOTA là khó hơn, nhưng có thể làm trong giai đoạn sau.

Bản đầu nên dùng BCE hoặc ranking loss.

---

## 12. Training procedure

Quy trình train:

```text
Step 1: Chạy detector trên video
Step 2: Chạy Kalman prediction cho tracks
Step 3: Extract ReID embeddings cho detections
Step 4: Sinh candidate pairs track-detection
Step 5: Tính motion cost và appearance cost
Step 6: Tạo feature vector cho từng pair
Step 7: Gán label bằng ground-truth ID
Step 8: Train Dynamic Weight Network
Step 9: Tích hợp vào tracker
Step 10: Evaluate trên benchmark
```

---

## 13. Evaluation metrics

Nên báo cáo:

```text
HOTA
MOTA
IDF1
ID Switches
FP
FN
AssA
DetA
FPS
```

Trong đó:

* **IDF1**: rất quan trọng vì module ảnh hưởng đến identity.
* **ID Switches**: cần giảm.
* **HOTA**: cân bằng detection và association.
* **FPS**: chứng minh module nhẹ.

---

## 14. Baseline cần so sánh

Nên so sánh với:

```text
DeepSORT
ByteTrack
BoT-SORT
StrongSORT
OC-SORT
DeepOCSORT
```

Ablation bắt buộc:

```text
Motion only
Appearance only
Fixed weight
Dynamic weight, no covariance
Dynamic weight, no detection confidence
Dynamic weight, no ReID features
Dynamic weight, full model
```

---

## 15. Đóng góp nghiên cứu tiềm năng

Có thể viết contribution như sau:

```text
We propose a dynamic motion-appearance fusion module for multi-object tracking, which estimates adaptive reliability weights for motion and appearance cues at each track-detection association step.
```

Các contribution chính:

1. Đề xuất module dynamic weight cho motion–appearance association.
2. Học trọng số theo từng track-detection pair thay vì dùng fixed weight.
3. Khai thác uncertainty từ Kalman Filter và độ tin cậy từ ReID.
4. Có thể tích hợp plug-and-play vào các tracker hiện có.
5. Cải thiện IDF1/HOTA và giảm ID Switches trên các benchmark khó.

---

## 16. Kế hoạch triển khai MVP

### Giai đoạn 1: Baseline

Chọn một tracker nền:

```text
BoT-SORT hoặc DeepSORT
```

Chạy baseline trên MOT17.

---

### Giai đoạn 2: Feature extraction

Extract:

```text
motion cost
appearance cost
track state features
detection features
```

Lưu thành file train:

```text
features.npy
labels.npy
```

---

### Giai đoạn 3: Train MLP

Train Dynamic Weight Network với BCE hoặc ranking loss.

---

### Giai đoạn 4: Tích hợp tracker

Thay công thức fixed fusion bằng:

```text
Cost_final = w_motion * Cost_motion + w_reid * Cost_reid
```

---

### Giai đoạn 5: Evaluate

Chạy trên:

```text
MOT17 validation
MOT20 validation
DanceTrack validation
```

Báo cáo:

```text
HOTA, IDF1, IDSw, MOTA, FPS
```

---

## 17. Rủi ro

Một số rủi ro chính:

1. Dynamic weight chỉ cải thiện nhỏ nếu baseline đã quá mạnh.
2. MLP có thể overfit dataset.
3. Nếu ReID model yếu, weight network cũng bị ảnh hưởng.
4. Nếu detector kém, lỗi detection có thể che mất lợi ích của association.
5. Nếu feature thiết kế chưa tốt, network chỉ học lại heuristic đơn giản.

---

## 18. Cách làm cho hướng này mạnh hơn

Để tăng khả năng publish, nên bổ sung:

```text
uncertainty-aware feature
```

Ví dụ:

```text
Kalman covariance
innovation error
ReID embedding variance
local crowd density
occlusion score
```

Ngoài ra nên chứng minh module:

```text
plug-and-play
```

tức là tích hợp được vào nhiều tracker khác nhau.

Ví dụ:

```text
BoT-SORT + Dynamic Weight
DeepSORT + Dynamic Weight
StrongSORT + Dynamic Weight
```

Nếu cả ba đều cải thiện, contribution sẽ mạnh hơn.

---

## 19. Kết luận

Hướng Dynamic Weight Motion–Appearance Fusion là khả thi để phát triển thành nghiên cứu. Phiên bản đơn giản chỉ là một module MLP điều chỉnh trọng số giữa Kalman Filter và ReID. Tuy nhiên, nếu mở rộng thành cơ chế **reliability-aware association**, sử dụng uncertainty của motion và confidence của appearance, hướng này có thể trở thành một đóng góp có giá trị trong Multi-Object Tracking.

Bản MVP nên bắt đầu từ:

```text
BoT-SORT + MOT17 + MLP dynamic weight
```

Sau đó mở rộng sang:

```text
MOT20, DanceTrack, SoccerNet
```

và chứng minh cải thiện ở:

```text
IDF1, HOTA, ID Switches
```
