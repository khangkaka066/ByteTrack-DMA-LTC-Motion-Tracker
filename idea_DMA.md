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

Dynamic Weight Network nhận feature vector mô tả độ tin cậy của motion và appearance.

Với mỗi cặp:

```text
track_i, detection_j
```

ta tạo input feature:

```text
x_ij = [
    motion features,
    appearance features,
    detection features,
    track state features
]
```

### 5.1. Motion features

Các feature liên quan đến Kalman Filter:

```text
IoU(predicted_bbox_i, detection_bbox_j)
1 - IoU
Mahalanobis distance
Kalman covariance mean
Kalman covariance trace
velocity magnitude
acceleration estimate
time_since_update
```

Ý nghĩa:

* covariance lớn → motion không chắc chắn
* time_since_update lớn → track đã mất lâu, Kalman kém tin cậy
* IoU thấp → prediction không khớp detection

---

### 5.2. Appearance features

Các feature liên quan đến ReID:

```text
cosine similarity(track_embedding_i, detection_embedding_j)
cosine distance
embedding norm
embedding variance
ReID confidence
```

Ý nghĩa:

* cosine distance thấp → hai object có appearance giống nhau
* embedding không ổn định → ReID kém tin cậy
* nhiều object giống nhau → appearance dễ gây ID switch

---

### 5.3. Detection features

Các feature từ detector:

```text
detection confidence
bbox area
bbox aspect ratio
bbox height
bbox width
```

Ý nghĩa:

* detection confidence thấp → detection có thể sai
* bbox nhỏ → ReID feature thường kém
* bbox bị méo hoặc quá nhỏ → appearance không đáng tin

---

### 5.4. Track history features

Các feature từ lịch sử track:

```text
track age
number of matched frames
number of missed frames
mean velocity
last matching confidence
track stability score
```

Ý nghĩa:

* track lâu và ổn định → motion có thể đáng tin hơn
* track mới tạo → chưa đủ lịch sử motion
* track mất nhiều frame → cần ReID nhiều hơn

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

Ví dụ kiến trúc:

```text
Input dim: 12–32
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
