# WORK_STATUS

## Mục tiêu hiện tại
Chốt gói nộp DeepWeeds fold 0 cho Nguyễn Minh Hiển (2A202602759) bằng kết quả đã chạy thật.

## Việc đã hoàn thành
- Đã đọc README, GUIDE, RUBRIC; hoàn thiện code từ starter trong thư mục bài nộp, giữ nguyên eval.py.
- Colab Tesla T4: xác minh MD5 của 17.509 ảnh, fold 0 train 10.501 / val 3.501 / test 3.507, không giao nhau hoặc thiếu ảnh.
- Test code 5/5, test gốc 38/38, smoke train/predict, pipeline ảnh thật và overfit 8 ảnh đã qua.
- Đã chạy B01–B05, T00–T04, I00–I04 trên val; chốt ConvNeXt Tiny + T02 + I04 trước test bằng final_lock.json.
- F01 đã chạy và xác minh test đủ 3 seed: macro-F1 0,972416 ± 0,002360; top-1 0,977854 ± 0,001855; ECE 0,006820 ± 0,000828.
- Baseline T00 đã chạy đủ 10 epoch, lưu dự đoán val/test và xác minh bằng eval.py score cho seed 0 và 1. Macro-F1 test trung bình tạm thời 0,967570 ± 0,000541.
- report.md có thiết lập, bảng kết quả, phân tích lỗi/ảnh, giới hạn và ghi rõ thiếu seed. results.xlsx gồm 7 sheet đã kiểm tra giá trị và ô lỗi. Có 15 đường cong, 3 confusion matrix, ảnh EDA/tradeoff/lỗi và dự đoán.
- Đã đóng gói submissions/2A202602759_NguyenMinhHien.zip: 64 file, 10 CSV dự đoán, 15 đường cong, không có checkpoint/dataset. Kiểm tra ZIP toàn vẹn, cú pháp 8 file Python, notebook JSON 21 ô và 7 sheet Excel đều qua.
- Đã khớp nhiệt độ trên val cho ba seed F01; không dùng test để chọn lại cấu hình.

## Việc đang làm
- Không có bước đang chạy. Gói nộp đã sẵn sàng với giới hạn baseline hai seed.

## File đã sửa/tạo
- submissions/2A202602759_NguyenMinhHien/: README.md, report.md, results.xlsx, final_lock.json, temperature_val.json, code/, predictions/, curves/, ảnh phân tích, eval_out/; submissions/2A202602759_NguyenMinhHien.zip.
- WORK_STATUS.md; .gitignore chỉ thêm .work/ vào ignore.
- .work/ có dữ liệu và script tạm, không đưa vào ZIP.

## Quyết định quan trọng
- Chọn backbone, training, inference bằng val; không thay đổi cấu hình sau khi mở test.
- Không tạo số liệu cho T00 seed 2. Người dùng chọn chốt bài với hai seed baseline sau khi Colab hết hạn mức GPU và Kaggle chưa bật GPU.
- Bảng và báo cáo chỉ nêu kết quả chạy thật; chênh lệch F01 ba seed so với T00 hai seed là tham khảo, không tính điểm I2.

## Lỗi/vấn đề còn tồn tại
- T00 seed 2 dừng ở epoch 4/10, không có CSV test; chưa chạy eval.py grade.
- RUBRIC.md điều kiện P4 yêu cầu prediction của chung kết và mốc ở mọi seed, nếu thiếu thì phần I có thể bị 0 điểm.
- Chưa chạy lại toàn notebook nguồn từ đầu trong phiên mới; các lượt thực nghiệm Colab trước đã lưu bằng chứng.

## Bước tiếp theo cần làm
1. Nếu có GPU sau này, chạy T00 seed 2 từ đầu, xuất val/test CSV, xác minh bằng eval.py score.
2. Khi đủ ba baseline seed, chạy eval.py grade, cập nhật results.xlsx và report.md.
