# DeepWeeds Lab Day 2 — 2A202602759 Nguyễn Minh Hiển

## Trạng thái

Phần `code/` đã hoàn thiện và kiểm thử bằng dữ liệu tổng hợp lẫn dữ liệu DeepWeeds thật trên Colab Tesla T4. Năm backbone B01–B05, T00–T04 và I00–I04 đã chạy; val được dùng để chốt `T02 + I04` trong `final_lock.json`. F01 và mốc T00 đều có dự đoán val/test thật cho đủ ba seed `0, 1, 2`; các CSV đã được `eval.py score` đối chiếu với nhãn gốc. T00 seed 2 được chạy lại từ đầu đủ 10 epoch trên Colab T4 sau khi phiên trước ngắt ở epoch 4. `eval.py grade` đã chạy: đề xuất 18/19 điểm cho các mục đã chấm của phần I; I4(a) chưa chấm được vì không có dự đoán test trước/sau temperature scaling. Đây là điểm tự chấm theo ngưỡng tạm thời, giảng viên xác nhận. `results.xlsx` và `report.md` chỉ dùng số đã kiểm chứng. Biểu đồ B/T seed 0 được dựng lại từ log đã lưu trong output notebook; T00 seed 2 có PNG gốc.

## Chạy lại trên Colab

1. Đưa **bản repo có `submissions/2A202602759_NguyenMinhHien/code/` này** lên Colab (clone sau khi đã đồng bộ code, hoặc tải bản repo lên). Không dùng bản `starter/` chưa hoàn thiện.
2. Bật GPU trong Colab. Mở `code/lab_day2.ipynb` và chạy từ trên xuống. Ô đầu cài các thư viện cần thiết và in phiên bản thực tế. Lượt đo trong báo cáo dùng PyTorch `2.11.0+cu130`, Tesla T4, 10 epoch và batch 64.
3. Ô dữ liệu tải ảnh DeepWeeds và các CSV fold 0. Kiểm tra MD5 và kết quả `dataset.check_split()` trước khi train.
4. Chạy EDA và kiểm tra pipeline, sau đó quét backbone, ablation và suy luận **chỉ trên val**. Mọi lần train dùng `train.run(Config(...))`.
5. Ô khóa cấu hình tự chọn theo val, ghi `final_lock.json` rồi mới cho phép phần chung kết và mốc chạy seed `0, 1, 2`. Cấu hình đã chốt cho lượt chạy trong báo cáo là ConvNeXt Tiny, tăng cường màu và suy luận ảnh gốc ở 256 px (`resolution256`). `train.run()` giữ val 224 px để chọn checkpoint khi huấn luyện và dùng loader 256 px riêng cho dự đoán val/test của I04.
6. Chạy `eval.py score` và `eval.py grade` như các ô cuối. Kết quả đối chiếu hiện tại nằm trong `eval_out/`; bảng và báo cáo chỉ dùng log/dự đoán chạy thật.

Test code cốt lõi trên Colab: `cd submissions/2A202602759_NguyenMinhHien/code && python -m unittest test_code -v`.

## Nguyên tắc dữ liệu

- Fold 0 cố định; train cập nhật trọng số, val chọn cấu hình và checkpoint, test chỉ chạy một lần sau khi chốt.
- CSV subset gốc chỉ có `Filename,Label`; `Species` được suy ra theo `Label` từ `labels.csv`. Một ảnh train (`20170714-110407-3.jpg`) mang Label 0 trong fold 0 và Label 1 trong `labels.csv`; code giữ nhãn fold 0 và ghi rõ sai khác này, không sửa CSV gốc.
- Không đưa ảnh, checkpoint hoặc thông tin cá nhân vào Git.
- Notebook Colab dùng để xác minh và chạy thí nghiệm thật: https://colab.research.google.com/drive/1Nu0ZfCysZZHUJ3ci6ZyntVxP744G5ojr. Lượt chạy lại T00 seed 2: https://colab.research.google.com/drive/1gb29lZpw1Cux2LSW_jW5w-YOm6CUEyVM. Các notebook Drive còn output lịch sử; bản quy trình đầy đủ, sạch nằm ở `code/lab_day2.ipynb`.
