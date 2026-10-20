---
title: 400VSL
emoji: 🤟
colorFrom: blue
colorTo: green
sdk: static
app_file: index.html
pinned: false
---

# 400VSL – bản web tĩnh

Web học 400 ký hiệu tiếng Việt (VSL400). Toàn bộ xử lý chạy trong trình duyệt của người dùng:
MediaPipe Tasks (WASM) trích khung xương, mô hình SDA-GCN chạy bằng ONNX Runtime Web.
Không có máy chủ, không gửi video đi đâu. Tiến độ học lưu trong trình duyệt (localStorage).

Chạy thử trên máy: trong thư mục này gõ `python -m http.server 8000` rồi mở http://127.0.0.1:8000
(phải mở qua http, không mở trực tiếp file index.html).
