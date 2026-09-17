# Task 1 — Refactor MoDANet Dataset Loader

## 1. Mục tiêu

Refactor phần đọc dữ liệu MoDANet để:

1. Đổi tên file:
   - từ `data/modanet_dataset.py`
   - thành `data/modanet_dataset_loader.py`

2. Đổi tên class:
   - từ `MoDANetDataset`
   - thành `MoDANetDatasetLoader`

3. Không giữ alias/tương thích ngược với tên cũ.
   - Không tạo `MoDANetDataset = MoDANetDatasetLoader`.
   - Không giữ lại file `data/modanet_dataset.py` sau khi refactor.

4. Cập nhật toàn bộ import/tham chiếu trong project đang dùng tên file/class cũ sang tên mới.

5. Biến `configs/modanet.yaml` thành **nguồn sự thật duy nhất** cho toàn bộ thông số dataset.

6. Loader **không tự đọc YAML**.
   - Code bên ngoài chịu trách nhiệm đọc YAML.
   - Loader chỉ nhận `dataset_config`, tức dictionary `cfg["dataset"]`.

7. Giữ lazy loading:
   - Không load toàn bộ dataset vào RAM.
   - Không quét và đọc toàn bộ 450,120 file `.mat` trong `__init__`.
   - Shape của từng sample chỉ được kiểm tra khi sample đó được đọc trong `__getitem__`.

8. Tạo test script đơn giản, không dùng pytest:
   - `store/test_modanet_dataset_loader.py`
   - Giữ lại file test này trong project sau khi hoàn thành.

---

## 2. Phạm vi công việc

### Được phép sửa

- `data/modanet_dataset.py`
- tạo `data/modanet_dataset_loader.py`
- `configs/modanet.yaml` nếu cần để bảo đảm schema/giá trị dataset hợp lệ và nhất quán
- các file trong project có import/tham chiếu trực tiếp tới:
  - `data.modanet_dataset`
  - `MoDANetDataset`
- tạo:
  - `store/test_modanet_dataset_loader.py`

### Không làm trong task này

Không nối `train/train_mamba_cnn.py` với pipeline YAML → dataset loader nếu hiện tại train script chưa sử dụng loader mới.

Cụ thể, **không refactor logic training**, loss, optimizer, model, checkpoint, batch size hay split train/validation trong task này.

Không thay đổi kiến trúc model.

Không thay đổi `mamba-main/`.

Không chuyển đổi toàn bộ dataset `.mat` sang format khác.

Không tạo cache/index dataset riêng nếu không thực sự cần cho tính đúng.

---

## 3. Kiến trúc mong muốn

Code bên ngoài đọc YAML:

```python
import yaml

with open("configs/modanet.yaml", "r") as f:
    cfg = yaml.safe_load(f)
```

Sau đó truyền nguyên khối dataset config:

```python
from data.modanet_dataset_loader import MoDANetDatasetLoader

dataset = MoDANetDatasetLoader(
    dataset_config=cfg["dataset"]
)
```

`MoDANetDatasetLoader` tuyệt đối không tự tìm hoặc tự mở `configs/modanet.yaml`.

Mục tiêu là tách trách nhiệm:

```text
YAML
  ↓
config layer / calling code
  ↓
dataset_config
  ↓
MoDANetDatasetLoader
  ↓
.mat → tensor + labels
```

---

## 4. Dataset config là nguồn sự thật duy nhất

Loader không được hard-code lại các thông số dataset sau nếu chúng đã có trong YAML.

Schema dataset mong muốn:

```yaml
dataset:
  root: /home/truongtn/MoDANet-dataset
  mat_variable: data

  num_samples: 450120

  snapshots: 1024
  antennas: 5
  iq_channels: 2

  num_mod_classes: 12
  modulations:
    - 16APSK
    - 16QAM
    - 4PAM
    - 64QAM
    - 8FSK
    - 8PSK
    - DSB-SC
    - LFM
    - PSK
    - QFSK
    - QPSK
    - SSB-SC

  doa_min: -60
  doa_max: 60
  doa_step: 1
  num_doa_classes: 121

loader:
  num_workers: 8
  pin_memory: false
  persistent_workers: true
```

Lưu ý:

- `loader:` là cấu hình của PyTorch `DataLoader`, không phải trách nhiệm nội tại của `MoDANetDatasetLoader`.
- `MoDANetDatasetLoader` chỉ nhận `cfg["dataset"]`.
- Không hard-code lại danh sách modulation trong file Python.
- Không hard-code `"data"` làm tên biến MATLAB.
- Không hard-code `(1024, 5, 2)`.
- Không hard-code `[-60, 60]`.
- Không hard-code `doa_label = doa_deg + 60`.

---

## 5. Constructor mới

API mong muốn:

```python
class MoDANetDatasetLoader(Dataset):
    def __init__(self, dataset_config):
        ...
```

Trong `__init__`, lấy các giá trị cần thiết từ dictionary:

```text
root
mat_variable
num_samples
snapshots
antennas
iq_channels
num_mod_classes
modulations
doa_min
doa_max
doa_step
num_doa_classes
```

Có thể lưu thành thuộc tính instance, ví dụ:

```python
self.root_dir
self.mat_variable
self.snapshots
self.antennas
self.iq_channels
self.modulations
self.mod_to_index
self.doa_min
self.doa_max
self.doa_step
self.num_doa_classes
```

Không bắt buộc đúng tên thuộc tính trên nếu có cách rõ ràng hơn, nhưng interface bên ngoài phải đúng yêu cầu.

---

## 6. Validation config — fail fast

Loader phải kiểm tra config ngay trong `__init__`.

### 6.1. Kiểm tra key bắt buộc

Nếu thiếu key cần thiết, raise exception rõ ràng, ưu tiên `KeyError` hoặc `ValueError` với message chỉ rõ key nào bị thiếu/sai.

### 6.2. Kiểm tra root

`root` phải tồn tại và là directory.

Nếu không:

```python
raise FileNotFoundError(...)
```

### 6.3. Kiểm tra modulation

Phải thỏa:

```python
num_mod_classes == len(modulations)
```

Danh sách modulation phải không rỗng và không có phần tử trùng.

Tạo mapping từ chính config:

```python
self.mod_to_index = {
    name: idx
    for idx, name in enumerate(self.modulations)
}
```

Mapping kỳ vọng hiện tại:

```text
0  -> 16APSK
1  -> 16QAM
2  -> 4PAM
3  -> 64QAM
4  -> 8FSK
5  -> 8PSK
6  -> DSB-SC
7  -> LFM
8  -> PSK
9  -> QFSK
10 -> QPSK
11 -> SSB-SC
```

### 6.4. Kiểm tra DOA config

Yêu cầu:

```text
doa_step > 0
doa_max >= doa_min
```

Số class theo config phải thỏa:

```python
expected_num_doa_classes = ((doa_max - doa_min) // doa_step) + 1
```

và:

```python
expected_num_doa_classes == num_doa_classes
```

Ngoài ra, khoảng phải chia hết theo `doa_step`:

```python
(doa_max - doa_min) % doa_step == 0
```

Nếu sai, raise `ValueError` rõ ràng.

### 6.5. Kiểm tra num_samples

Loader vẫn có thể dùng:

```python
self.files = sorted(
    self.root_dir.glob("SNR*dB/*/*/*.mat")
)
```

Sau khi lập danh sách file, kiểm tra:

```python
len(self.files) == num_samples
```

Với dataset hiện tại phải là:

```text
450120
```

Nếu không khớp, raise `ValueError` hoặc `RuntimeError` với:
- expected
- actual
- dataset root

Mục đích là phát hiện dataset thiếu file hoặc config sai.

### 6.6. Không quét nội dung toàn bộ file lúc init

Không gọi `loadmat` cho toàn bộ 450,120 file trong `__init__`.

Chỉ lập danh sách path và kiểm tra số lượng file.

---

## 7. Logic đọc sample

Mỗi file `.mat` có cấu trúc dữ liệu hiện tại:

```text
data shape = (1024, 5, 2)
dtype = float32
```

Nhưng loader không được hard-code các số này.

Expected shape phải được tạo từ config:

```python
expected_shape = (
    self.snapshots,
    self.antennas,
    self.iq_channels,
)
```

Ví dụ config hiện tại:

```text
(1024, 5, 2)
```

Đọc biến MATLAB bằng tên trong:

```python
self.mat_variable
```

Ví dụ hiện tại là:

```text
data
```

Nếu biến không tồn tại, raise `KeyError` và ghi rõ path file.

Nếu shape khác expected shape, raise `ValueError` và ghi rõ:
- actual shape
- expected shape
- file path

---

## 8. Chuyển shape

Dữ liệu gốc:

```text
(T, M, IQ)
=
(1024, 5, 2)
```

Output tensor cho model:

```text
(IQ, M, T)
=
(2, 5, 1024)
```

Có thể tiếp tục dùng:

```python
x = np.transpose(x, (2, 1, 0))
x = np.ascontiguousarray(x, dtype=np.float32)
x = torch.from_numpy(x)
```

Không thay đổi nội dung số liệu, chỉ đổi thứ tự trục và bảo đảm `float32` contiguous tensor.

---

## 9. Parse metadata từ path

Dataset có dạng:

```text
root/
  SNR-07dB/
    QPSK/
      +51/
        data00048308.mat
```

Từ path phải suy ra:

```text
snr_db = -7
modulation_name = QPSK
doa_deg = +51
```

Giữ logic regex SNR tương đương hiện tại:

```python
r"SNR([+-]\d+)dB"
```

Nếu không parse được, raise error rõ ràng.

---

## 10. Mapping modulation

Không còn constant toàn cục:

```python
MODULATIONS = [...]
MOD_TO_INDEX = {...}
```

Mapping phải xuất phát hoàn toàn từ:

```python
dataset_config["modulations"]
```

Nếu tên modulation từ path không thuộc config, raise `ValueError`.

---

## 11. Mapping DOA tổng quát

Không dùng:

```python
doa_label = doa_deg + 60
```

Dùng config:

```python
doa_label = (doa_deg - self.doa_min) // self.doa_step
```

Trước khi tính label phải kiểm tra:

```python
self.doa_min <= doa_deg <= self.doa_max
```

và góc nằm đúng trên grid:

```python
(doa_deg - self.doa_min) % self.doa_step == 0
```

Nếu không đúng, raise `ValueError`.

Với config hiện tại phải có:

```text
-60° -> label 0
  0° -> label 60
+60° -> label 120
```

---

## 12. Output của `__getitem__`

Giữ nguyên đủ 6 field:

```python
return {
    "x": x,
    "mod_label": torch.tensor(mod_label, dtype=torch.long),
    "doa_label": torch.tensor(doa_label, dtype=torch.long),
    "doa_deg": doa_deg,
    "snr_db": snr_db,
    "path": str(file_path),
}
```

Không tối giản output trong task này.

Kỳ vọng:

```text
x            -> torch.Tensor, float32, shape (2, 5, 1024)
mod_label    -> scalar torch.long
doa_label    -> scalar torch.long
doa_deg      -> int
snr_db       -> int
path         -> str
```

---

## 13. Đổi tên file và class

Thực hiện:

```text
data/modanet_dataset.py
→
data/modanet_dataset_loader.py
```

và:

```text
MoDANetDataset
→
MoDANetDatasetLoader
```

Sau refactor:

- Không còn file `data/modanet_dataset.py`.
- Không còn định nghĩa class `MoDANetDataset`.
- Không giữ alias compatibility.
- Search toàn repository và cập nhật mọi import/reference thật sự đang tồn tại.

Ví dụ:

```python
from data.modanet_dataset import MoDANetDataset
```

phải thành:

```python
from data.modanet_dataset_loader import MoDANetDatasetLoader
```

Không tạo import mới trong `train/train_mamba_cnn.py` nếu file đó hiện tại chưa dùng dataset loader.

---

## 14. Test script phải tạo

Tạo:

```text
store/test_modanet_dataset_loader.py
```

Không dùng `pytest`.

Script phải chạy trực tiếp:

```bash
python3 store/test_modanet_dataset_loader.py
```

và exit code phải là `0` khi mọi test pass.

Nếu test fail:
- raise `AssertionError` hoặc exception rõ ràng
- không nuốt lỗi
- không chỉ `print` rồi tiếp tục

---

## 15. Nội dung test bắt buộc

### 15.1. Đọc config

Test tự đọc:

```text
configs/modanet.yaml
```

bằng `yaml.safe_load`.

Sau đó:

```python
dataset_cfg = cfg["dataset"]
```

và khởi tạo:

```python
dataset = MoDANetDatasetLoader(
    dataset_config=dataset_cfg
)
```

### 15.2. Kiểm tra số sample

```python
assert len(dataset) == 450120
```

Tốt hơn là so với config:

```python
assert len(dataset) == dataset_cfg["num_samples"]
```

### 15.3. Kiểm tra keys output

Một sample phải có đúng các field cần thiết:

```text
x
mod_label
doa_label
doa_deg
snr_db
path
```

Không yêu cầu dictionary chỉ có duy nhất 6 key nếu có lý do tốt để thêm metadata, nhưng trong task này không cần thêm field mới.

### 15.4. Kiểm tra shape và dtype

```text
x.shape == (2, 5, 1024)
x.dtype == torch.float32
```

Nên suy ra expected shape từ config thay vì hard-code trong logic test:

```python
expected_output_shape = (
    dataset_cfg["iq_channels"],
    dataset_cfg["antennas"],
    dataset_cfg["snapshots"],
)
```

### 15.5. Kiểm tra type

- `x` là `torch.Tensor`
- `mod_label` là scalar tensor `torch.long`
- `doa_label` là scalar tensor `torch.long`
- `doa_deg` là integer-compatible
- `snr_db` là integer-compatible
- `path` là `str`

### 15.6. Kiểm tra mapping 12 modulation

Xác nhận mapping của loader đúng theo thứ tự config.

Kỳ vọng hiện tại:

```text
16APSK -> 0
16QAM  -> 1
4PAM   -> 2
64QAM  -> 3
8FSK   -> 4
8PSK   -> 5
DSB-SC -> 6
LFM    -> 7
PSK    -> 8
QFSK   -> 9
QPSK   -> 10
SSB-SC -> 11
```

Không cần load đủ 12 sample chỉ để test mapping nếu mapping instance có thể kiểm tra trực tiếp an toàn.

### 15.7. Kiểm tra DOA mapping biên

Phải xác nhận logic:

```text
-60° -> 0
0°   -> 60
+60° -> 120
```

Nên test công thức/method nội bộ hợp lý hoặc tìm sample thực nếu tiện.

Không được viết test phụ thuộc nguy hiểm vào thứ tự `sorted()` nếu có cách test ổn định hơn.

### 15.8. Kiểm tra sample thật 16APSK / 0°

Dataset hiện có sample dạng:

```text
SNR+00dB/16APSK/+00/...
```

Test nên xác nhận ít nhất một sample thực có:

```text
mod_label == 0
doa_deg == 0
doa_label == 60
```

Không bắt buộc hard-code filename cụ thể nếu có thể tìm sample theo path pattern.

### 15.9. Kiểm tra DataLoader batch

Tạo một PyTorch DataLoader nhỏ, ví dụ:

```python
DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
)
```

Dùng `num_workers=0` trong regression test để:
- giảm phụ thuộc multiprocessing
- test đơn giản, ổn định

Lấy một batch và kiểm tra:

```text
batch["x"].shape == (4, 2, 5, 1024)
batch["mod_label"].shape == (4,)
batch["doa_label"].shape == (4,)
```

Expected shape nên suy ra từ config.

---

## 16. Không benchmark hiệu năng trong regression test

Không đưa benchmark `num_workers=8` vào `store/test_modanet_dataset_loader.py`.

Các benchmark đã thực hiện trên laptop cho thấy:

```text
num_workers=0  -> ~1754 samples/s
num_workers=2  -> ~3051 samples/s
num_workers=4  -> ~5364 samples/s
num_workers=8  -> ~7559 samples/s
num_workers=10 -> ~7764 samples/s
```

Máy có 12 logical CPU threads.

Quyết định hiện tại cho laptop:

```yaml
loader:
  num_workers: 8
  pin_memory: false
  persistent_workers: true
```

Các giá trị này thuộc config DataLoader và có thể benchmark lại trên A100/server.

Không nhúng chúng vào class dataset loader.

---

## 17. Hành vi không được thay đổi

Sau refactor, các đặc tính sau phải giữ nguyên:

1. Dataset được lazy-load từng `.mat`.
2. File `.mat` gốc không bị sửa.
3. Không copy/chuyển đổi 18 GB dataset.
4. Dữ liệu model nhận vẫn là:
   ```text
   (2, 5, 1024)
   ```
   cho mỗi sample với config hiện tại.
5. Batch vẫn có dạng:
   ```text
   (B, 2, 5, 1024)
   ```
6. Nhãn modulation vẫn là integer class index.
7. Nhãn DOA vẫn là integer class index `0..120`.
8. Giữ metadata `doa_deg`, `snr_db`, `path`.

---

## 18. Yêu cầu về chất lượng code

- Code rõ ràng, dễ đọc.
- Không over-engineer.
- Không thêm dependency mới nếu không cần thiết.
- Dùng `pathlib.Path`.
- Giữ `numpy`, `torch`, `scipy.io.loadmat`.
- Error message phải đủ thông tin để debug.
- Không swallow exception.
- Không dùng broad `except Exception` chỉ để bỏ qua sample lỗi.
- Không âm thầm sửa config sai.
- Không thay đổi unrelated files.
- Không format/rewrite toàn repo chỉ vì task nhỏ này.

---

## 19. Trình tự Codex nên thực hiện

1. Đọc trạng thái hiện tại của:
   - `data/modanet_dataset.py`
   - `configs/modanet.yaml`
   - toàn repo để tìm reference tới tên cũ.

2. Xác nhận schema config hiện tại.

3. Rename:
   ```text
   data/modanet_dataset.py
   → data/modanet_dataset_loader.py
   ```

4. Rename class:
   ```text
   MoDANetDataset
   → MoDANetDatasetLoader
   ```

5. Refactor constructor nhận:
   ```python
   dataset_config
   ```

6. Xóa hard-code dataset metadata khỏi Python.

7. Thêm config validation.

8. Tổng quát hóa DOA mapping.

9. Cập nhật import/reference tên cũ trên repository.

10. Tạo:
    ```text
    store/test_modanet_dataset_loader.py
    ```

11. Chạy:
    ```bash
    python3 store/test_modanet_dataset_loader.py
    ```

12. Chạy thêm:
    ```bash
    python3 -m pip check
    ```
    nếu environment hiện tại đã được activate.

13. Search lại để bảo đảm không còn reference không chủ đích tới:
    ```text
    modanet_dataset
    MoDANetDataset
    ```
    Lưu ý `modanet_dataset_loader` chứa chuỗi `modanet_dataset`, vì vậy search phải phân biệt tên cũ chính xác.

14. Báo cáo ngắn gọn:
    - files changed
    - tests run
    - test result
    - bất kỳ vấn đề nào còn lại

---

## 20. Acceptance criteria

Task chỉ được coi là hoàn thành nếu tất cả điều sau đúng:

- [ ] Có `data/modanet_dataset_loader.py`.
- [ ] Không còn `data/modanet_dataset.py`.
- [ ] Class chính là `MoDANetDatasetLoader`.
- [ ] Không có alias `MoDANetDataset`.
- [ ] Loader nhận `dataset_config`.
- [ ] Loader không tự đọc YAML.
- [ ] `configs/modanet.yaml` là nguồn sự thật duy nhất cho dataset metadata.
- [ ] Không còn hard-code modulation list trong loader.
- [ ] Không còn hard-code `"data"` trong logic đọc `.mat`.
- [ ] Expected input shape được suy ra từ config.
- [ ] DOA range/step/classes được suy ra và validation từ config.
- [ ] DOA label dùng công thức tổng quát theo `doa_min` và `doa_step`.
- [ ] `num_mod_classes == len(modulations)` được validate.
- [ ] `num_doa_classes` được validate.
- [ ] `len(files) == num_samples` được validate.
- [ ] Shape từng `.mat` được kiểm tra lazy trong `__getitem__`.
- [ ] Output vẫn có đủ `x`, `mod_label`, `doa_label`, `doa_deg`, `snr_db`, `path`.
- [ ] Tạo và giữ `store/test_modanet_dataset_loader.py`.
- [ ] Test script pass.
- [ ] Batch test pass.
- [ ] Không thay đổi training pipeline trong task này.
- [ ] Không thay đổi model/Mamba source.
- [ ] Không thay đổi dataset gốc.

---

## 21. Kết quả mong đợi sau task

Cách dùng tối thiểu:

```python
import yaml
from data.modanet_dataset_loader import MoDANetDatasetLoader

with open("configs/modanet.yaml", "r") as f:
    cfg = yaml.safe_load(f)

dataset = MoDANetDatasetLoader(
    dataset_config=cfg["dataset"]
)

sample = dataset[0]

print(len(dataset))
print(sample["x"].shape)
print(sample["mod_label"])
print(sample["doa_label"])
```

Với dataset/config hiện tại, kết quả phải tương thích:

```text
len(dataset) = 450120
sample["x"].shape = torch.Size([2, 5, 1024])
```

Một sample thực tại:

```text
SNR+00dB/16APSK/+00/...
```

phải ánh xạ:

```text
16APSK -> mod_label 0
0°     -> doa_label 60
```

---

## 22. Nguyên tắc quan trọng

**Không tối ưu thêm ngoài phạm vi task.**

Mục tiêu của task 1 là làm cho loader:
- tên rõ ràng,
- config-driven,
- validation tốt,
- test được,
- không lặp nguồn sự thật,
- vẫn giữ đúng hành vi dữ liệu hiện tại.

Sau khi task này pass mới tiếp tục sang việc tích hợp loader/config với training pipeline.
