# DeepWeeds fold 0: backbone, công thức huấn luyện và suy luận

**Nguyễn Minh Hiển — 2A202602759**

## 1. Tóm tắt

Đã kiểm tra pipeline trên dữ liệu DeepWeeds thật và so sánh năm backbone với cùng cấu hình huấn luyện 10 epoch, seed 0 trên Tesla T4. ConvNeXt Tiny đạt macro-F1 val cao nhất trong lượt quét backbone: 0,9680. Sau ba ablation đơn và một tổ hợp, T02 được chọn bằng val. Trong năm phương pháp suy luận, I04 (ảnh gốc 256 px) được chọn theo quy tắc chất lượng trong 0,005 của điểm cao nhất rồi ưu tiên p95 thấp. Cấu hình F01 đạt macro-F1 test **0,9724 ± 0,0024** và top-1 test **0,9779 ± 0,0019** qua ba seed. Mốc T00 có hai seed hoàn tất, macro-F1 test 0,9676 ± 0,0005; kết quả này chỉ tham khảo vì rubric yêu cầu ba seed cho so sánh cuối.

## 2. Dữ liệu và thiết lập

Sử dụng fold 0 của DeepWeeds với 9 lớp. Bộ ảnh từ Zenodo có MD5 `b7b30f96d466fba86016aa5a26606e0f`, trùng checksum trong hướng dẫn. Các file chia tập của tác giả gồm 10.501 ảnh train, 3.501 ảnh val và 3.507 ảnh test. Ba tập không trùng tên ảnh, hợp đủ 17.509 ảnh và không thiếu tệp; ảnh được kiểm tra là RGB 256 × 256 pixel.

| Label | Lớp | Train | Val | Test |
|---:|---|---:|---:|---:|
| 0 | Chinee Apple | 675 | 225 | 226 |
| 1 | Lantana | 637 | 213 | 213 |
| 2 | Parkinsonia | 618 | 206 | 207 |
| 3 | Parthenium | 613 | 204 | 205 |
| 4 | Prickly Acacia | 637 | 212 | 213 |
| 5 | Rubber Vine | 605 | 202 | 202 |
| 6 | Siam Weed | 644 | 215 | 215 |
| 7 | Snake Weed | 609 | 203 | 204 |
| 8 | Negatives | 5.463 | 1.821 | 1.822 |

Tỉ lệ giữa lớp nhiều nhất và ít nhất là 9,0248; `Negatives` chiếm 9.106 ảnh, khớp số tham khảo trong README. Macro-F1 trên 9 lớp là chỉ số chọn mô hình vì accuracy có thể che lỗi ở lớp hiếm. Dữ liệu đánh giá dùng resize cạnh ngắn 256 rồi center crop 224; train dùng random resized crop 224 và lật ngang. Cả hai chuẩn hóa theo mean/std ImageNet. Công thức nền: trọng số tiền huấn luyện, AdamW, LR backbone `1e-4`, LR head `1e-3`, weight decay `0,05` trừ norm/bias, warmup 1 epoch rồi cosine theo bước, batch 64, AMP và 10 epoch. Checkpoint được chọn bằng macro-F1 val; test chưa được dùng trong lượt quét.

CSV của tác giả có một điểm không nhất quán: `20170714-110407-3.jpg` có Label 0 trong `train_subset0.csv` nhưng Label 1 trong `labels.csv`. Code giữ nhãn của fold 0 cho huấn luyện/đánh giá và suy ra tên loài theo Label. Không sửa dữ liệu gốc.

Kiểm tra pipeline trước sweep: CE ban đầu trên ResNet18 khởi tạo là 2,1964, gần `ln(9) = 2,1972`. Khi cố ý overfit 8 ảnh thật trong 25 bước, loss xuống 0,0073 và accuracy đạt 1,0. Năm unit test bổ sung, 38 test gốc và smoke test train/suy luận đều qua trên Colab. Biểu đồ phân bố lớp ở `eda_counts.png` được dựng từ bảng đếm do Colab in ra. Ảnh ví dụ augmentation nằm trong output notebook Colab, chưa được đưa về thư mục nộp.

## 3. So sánh backbone trên val

Mỗi dòng dùng seed 0, fold 0, 10 epoch, batch 64 và cùng công thức nền. Tag trọng số được lưu vì nguồn tiền huấn luyện khác nhau có thể ảnh hưởng thứ hạng. Bảng dùng checkpoint có macro-F1 val cao nhất trong 10 epoch.

| ID | Backbone | Tag trọng số | Epoch tốt nhất | Params (M) | GMAC do fvcore báo | Macro-F1 val | Top-1 val | Train (giây/epoch) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| B01 | ResNet50 | `a1_in1k` | 9 | 23,526 | 4,109 | 0,7937 | 0,8495 | 45,127 |
| B02 | ResNeXt50-32x4d | `a1h_in1k` | 4 | 22,998 | 4,257 | 0,7401 | 0,8089 | 56,424 |
| B03 | ConvNeXt Tiny | `in12k_ft_in1k` | 9 | 27,827 | 4,470 | 0,9680 | 0,9766 | 51,402 |
| B04 | DeiT Small | `fb_in1k` | 10 | 21,669 | 4,250* | 0,9477 | 0,9634 | 41,477 |
| B05 | EfficientNet-B0 | `ra_in1k` | 10 | 4,019 | 0,398 | 0,7653 | 0,8246 | 42,814 |

*`fvcore` cảnh báo bỏ qua một số toán tử attention/GELU khi đếm B04, nên GMAC của B04 có thể bị đánh giá thấp. B03 dùng tag tiền huấn luyện `in12k_ft_in1k`, khác các tag còn lại, nên chênh lệch trong bảng không tách riêng tác động của kiến trúc và nguồn trọng số. Đây là lượt quét một seed; chưa thể dùng chênh lệch nhỏ để khẳng định hơn kém ổn định. B02 đạt đỉnh ở epoch 4 rồi train loss tiếp tục giảm mà điểm val không vượt đỉnh, dấu hiệu quá khớp theo lịch này.

Độ trễ forward batch 1 đo trên Tesla T4, fp32, ảnh 224 × 224, 10 warmup và 50 lượt đo, đồng bộ CUDA; không gồm đọc ảnh hay chuẩn hóa:

| ID | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---:|---:|---:|
| B01 | 6,343 | 6,718 | 7,882 |
| B02 | 8,882 | 14,472 | 15,460 |
| B03 | 6,104 | 6,766 | 7,004 |
| B04 | 5,134 | 5,498 | 5,584 |
| B05 | 8,392 | 12,535 | 14,916 |

Độ trễ thực đo không đi cùng thứ tự GMAC: B05 có GMAC thấp nhất nhưng p95 cao hơn B03 trên T4 ở batch 1. Đây là phép đo forward riêng, không đại diện thời gian xử lý ảnh đầu cuối.

Biểu đồ macro-F1 val theo p95 của năm backbone ở `backbone_tradeoff.png`; đường cong loss và macro-F1 từng epoch ở `curves/`.

## 4. Công thức huấn luyện

Chạy T00–T04 trên ConvNeXt Tiny được chọn từ val. Các so sánh đơn yếu tố gồm khởi tạo đóng băng, tăng cường màu và label smoothing; T04 kết hợp hai thay đổi có macro-F1 val cao nhất trong ba ablation đơn.

| ID | Khác T00 | Macro-F1 val | Top-1 val | Δ macro-F1 val | Train (giây/epoch) |
|---|---|---:|---:|---:|---:|
| T00 | Công thức nền, fine-tune toàn bộ | 0,9680 | 0,9766 | 0,0000 | 50,624 |
| T01 | Đóng băng backbone, chỉ train head | 0,8453 | 0,8777 | −0,1227 | 38,245 |
| T02 | Tăng cường màu | 0,9692 | 0,9757 | +0,0012 | 103,286 |
| T03 | Label smoothing 0,1 | 0,9663 | 0,9746 | −0,0017 | 50,419 |
| T04 | Tăng cường màu + label smoothing 0,1 | 0,9686 | 0,9757 | +0,0006 | 102,662 |

T01 tiết kiệm 12,378 giây/epoch so với T00 nhưng giảm macro-F1 val 0,1227 trên cùng seed. T02 và T04 chỉ hơn T00 lần lượt 0,0012 và 0,0006 macro-F1, trong khi thời gian train mỗi epoch khoảng gấp đôi. Chưa có độ lệch chuẩn qua seed, nên không kết luận được các chênh lệch nhỏ này là cải thiện ổn định. T03 thấp hơn T00 0,0017 trên một seed, cũng chưa đủ để kết luận label smoothing gây hại. T04 là tổ hợp của hai ablation đơn có macro-F1 val cao nhất (T02 và T03), không được xem là thí nghiệm đơn yếu tố.

## 5. Phương pháp suy luận và độ trễ

Các phương pháp được so sánh trên checkpoint khôi phục T02R, cùng công thức T02 và seed 0. T02R được huấn luyện lại sau khi phiên Colab trước xóa checkpoint; macro-F1 val I00 của checkpoint này là 0,964695, khác lượt T02 ban đầu 0,969200. I00 dùng center crop 224; I01 trung bình xác suất hai view gốc và lật ngang; I02 trung bình xác suất view gốc và crop giữa; I03 trung bình logit hai view gốc và lật ngang; I04 đọc ảnh gốc ở 256 px. I04 **không** dùng phép phóng ảnh đã crop 224 px. Dự đoán trên val không đọc test.

| ID | K | Macro-F1 val | Top-1 val | ECE val | p50 / p95 / p99 batch 1 (ms) | p95 so với I00 |
|---|---:|---:|---:|---:|---|---:|
| I00 | 1 | 0,9647 | 0,9729 | 0,0142 | 6,264 / 7,048 / 7,944 | 1,00× |
| I01 | 2 | 0,9658 | 0,9740 | 0,0096 | 12,421 / 14,854 / 18,610 | 2,11× |
| I02 | 2 | 0,9706 | 0,9774 | 0,0082 | 11,845 / 13,830 / 15,806 | 1,96× |
| I03 | 2 | 0,9657 | 0,9737 | 0,0123 | 11,840 / 13,612 / 14,946 | 1,93× |
| I04 | 1 | 0,9686 | 0,9751 | 0,0117 | 6,660 / 8,711 / 10,246 | 1,24× |

Độ trễ GPU fp32 đo trên Tesla T4, 10 warmup và 50 lượt đo có đồng bộ CUDA; không gồm đọc ảnh, center crop, resize hay chuẩn hóa. Ở batch 32, p95 lần lượt I00–I04 là 131,863; 261,824; 264,257; 265,831; 169,716 ms. I02 có macro-F1 val cao nhất, tăng 0,0059 so với I00 và tốn gần gấp đôi p95. I04 tăng 0,0039 với p95 tăng 1,24 lần, vẫn dưới ngân sách 100 ms của rubric. ECE của I02 và I04 thấp hơn I00 trong lượt val này; các chênh lệch nhỏ vẫn cần kiểm tra qua seed. Biểu đồ đánh đổi ở `inference_tradeoff.png`.

Theo quy tắc ghi trong `final_lock.json`, các phương pháp có macro-F1 cách điểm cao nhất không quá 0,005 được xét tiếp; trong nhóm này I04 có p95 batch 1 thấp nhất, nên được chọn trước khi mở test.

Để kiểm tra hiệu chuẩn, sau khi khóa cấu hình tôi khớp một nhiệt độ riêng cho mỗi seed F01 bằng cách cực tiểu NLL **chỉ trên val**, rồi tính lại ECE val từ cùng 3.501 ảnh. Xác suất đã lưu cho phép tính lại `softmax(log(p)/T)`; code và giá trị đầy đủ ở `code/temperature_val.py` và `temperature_val.json`. Đây là phân tích hậu nghiệm trên val, không thay đổi dự đoán test hay cấu hình F01 đã khóa.

| Seed | T khớp trên val | ECE val trước | ECE val sau | NLL val trước → sau |
|---:|---:|---:|---:|---:|
| 0 | 1,3617 | 0,01174 | 0,00537 | 0,09402 → 0,08574 |
| 1 | 1,2230 | 0,00901 | 0,00377 | 0,08824 → 0,08473 |
| 2 | 1,2284 | 0,00701 | 0,00392 | 0,07486 → 0,07172 |

Macro-F1 val không đổi vì chia logit cho nhiệt độ dương giữ nguyên lớp argmax. ECE sau khớp đo trên chính val dùng để chọn T, vì vậy chưa chứng minh hiệu chuẩn sẽ cải thiện trên tập mới; mục I4(a) của rubric chưa có điểm test sau hiệu chuẩn.

## 6. Cấu hình cuối và test

Đã ghi `final_lock.json` trên Colab trước khi mở test: ConvNeXt Tiny, T02 tăng cường màu, 10 epoch, batch 64, I04 ảnh gốc 256 px, seed 0/1/2. F01 seed 0 dùng lại checkpoint T02R, chính là lượt huấn luyện cùng cấu hình và seed 0, rồi dự đoán test một lần theo I04. Seed 1–2 được huấn luyện độc lập cùng cấu hình đã khóa. CSV val/test của cả ba seed đã lưu trong `predictions/`; `eval.py score` xác nhận 3.501/3.507 hàng, tên ảnh/nhãn đúng CSV fold 0, `y_pred = argmax(p0…p8)` và xác suất hợp lệ. Mốc T00 seed 0–1 đã chạy đủ 10 epoch và có CSV val/test; seed 2 bị Colab ngắt ở epoch 4/10 do hết hạn mức GPU. Kaggle không mở GPU cho tài khoản hiện tại. Theo lựa chọn của người nộp, báo cáo chốt với hai seed mốc, **không** điền số liệu seed 2 và chưa tự chấm I2 bằng `eval.py grade`.

| Cấu hình | Seed | Macro-F1 val | Macro-F1 test | Top-1 test | ECE test |
|---|---:|---:|---:|---:|---:|
| F01: T02 + I04 | 0 | 0,968551 | 0,973079 | 0,977759 | 0,007744 |
| F01: T02 + I04 | 1 | 0,967357 | 0,969796 | 0,976048 | 0,006569 |
| F01: T02 + I04 | 2 | 0,974302 | 0,974374 | 0,979755 | 0,006146 |
| **F01 mean ± std** | **3 seed** | **0,970070 ± 0,003713** | **0,972416 ± 0,002360** | **0,977854 ± 0,001855** | **0,006820 ± 0,000828** |
| T00: nền + I00 | 0 | 0,968072 | 0,967188 | 0,974337 | 0,011689 |
| T00: nền + I00 | 1 | 0,963162 | 0,967952 | 0,975192 | 0,009698 |
| **T00 tham khảo mean ± std** | **2 seed** | **0,965617 ± 0,003472** | **0,967570 ± 0,000541** | **0,974765 ± 0,000605** | **0,010694 ± 0,001408** |

Chênh lệch macro-F1 test giữa trung bình F01 ba seed và T00 hai seed là +0,004846. Hai nhóm **không cùng số seed**, vì vậy đây là đối chiếu sơ bộ và không đủ điều kiện chấm I2. Không chọn lại cấu hình từ kết quả test này.

Chỉ số theo lớp trên test dưới đây là trung bình của ba seed F01, tính từ các CSV dự đoán bằng `eval.py`; số ảnh mỗi lớp là số ảnh trong fold 0 test.

| Lớp | Số ảnh test | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Chinee apple | 226 | 0,9691 | 0,9233 | 0,9456 |
| Lantana | 213 | 0,9858 | 0,9703 | 0,9779 |
| Parkinsonia | 207 | 0,9824 | 0,9823 | 0,9823 |
| Parthenium | 205 | 0,9966 | 0,9610 | 0,9784 |
| Prickly acacia | 213 | 0,9496 | 0,9734 | 0,9614 |
| Rubber vine | 202 | 0,9949 | 0,9670 | 0,9807 |
| Siam weed | 215 | 0,9891 | 0,9798 | 0,9844 |
| Snake weed | 204 | 0,9622 | 0,9510 | 0,9565 |
| Negative | 1.822 | 0,9775 | 0,9914 | 0,9844 |

Recall của Chinee apple và Snake weed lần lượt là 0,9233 và 0,9510, đều vượt mốc tham khảo 0,885 và 0,888 của bài báo; hai kết quả không hoàn toàn cùng điều kiện huấn luyện hoặc cách tính accuracy. Ma trận nhầm lẫn theo từng seed có trong `confusion_matrix_F01_seed{0,1,2}_test.png`.

`error_examples_F01_seed0.jpg` ghép sáu ảnh test mà seed 0 dự đoán sai với độ tin cậy cao nhất, lấy trực tiếp từ ảnh và CSV dự đoán thật. Năm ảnh thuộc lớp cỏ dại được dự đoán là `Negative`; một ảnh `Negative` được dự đoán là Siam Weed. Trong các ảnh này, cây mục tiêu thường nhỏ hoặc lẫn với nhiều thân/lá và nền đất, nên cần xem đây là các ca khó để kiểm tra định tính, không suy diễn chúng đại diện cho toàn bộ lỗi. Độ tin cậy rất cao ở cả sáu ca cho thấy vẫn có lỗi quá tự tin dù ECE trung bình thấp.

Trong ma trận nhầm lẫn seed 0, Chinee apple bị đoán thành Snake weed 1 lần và Snake weed bị đoán thành Chinee apple 2 lần. `error_pair_chinee_snake_seed0.jpg` chứa đúng ba ảnh này. Một ảnh ngược sáng, hai ảnh có cây xen dày với bóng tối; giả thuyết cần kiểm tra là chi tiết lá phân biệt hai lớp bị che hoặc quá nhỏ trong ảnh toàn cảnh. Ba ca này chưa đủ để chứng minh nguyên nhân, nhưng chỉ ra nhóm ảnh nên gắn nhãn lại để kiểm tra chất lượng dữ liệu và thử crop có giám sát ở nghiên cứu sau.

## 7. Kết luận và khuyến nghị

Lượt quét backbone một seed cho thấy ConvNeXt Tiny với trọng số `in12k_ft_in1k` vượt các backbone còn lại trong thiết lập này, nhưng không tách được ảnh hưởng kiến trúc khỏi nguồn tiền huấn luyện. T02 hơn T00 rất ít ở val gốc và tăng thời gian train đáng kể. I04 cho điểm val gần cao nhất với chi phí suy luận thấp hơn các phương pháp hai view. F01 đã có đủ ba seed và p95 batch 1 là 8,711 ms, thấp hơn ngân sách 100 ms của rubric. Chênh lệch test so với T00 mới là đối chiếu sơ bộ vì baseline còn thiếu seed 2.

## 8. Hạn chế

Hiện các kết quả backbone chỉ có một seed, một fold và chia ảnh ngẫu nhiên thay vì chia theo địa điểm. Baseline cuối mới có hai seed, nên phần so sánh cải thiện và chấm I2 chưa đạt yêu cầu. Điểm test cuối có thể lạc quan khi triển khai ở địa điểm hoặc mùa khác. Tag tiền huấn luyện khác nhau và phép đếm GMAC của DeiT có giới hạn đã nêu ở trên.

## 9. Phụ lục và tái lập

Notebook Colab có output từ các lần chạy thật: <https://colab.research.google.com/drive/1Nu0ZfCysZZHUJ3ci6ZyntVxP744G5ojr>. Phiên Colab đầu đã hết thời gian và xóa tệp `/content`. Biểu đồ B01–B05/T00–T04 trong `curves/` được dựng lại từ log từng epoch đã lưu trong notebook (loss/F1 làm tròn bốn chữ số, thời gian một chữ số), nên không phải PNG gốc của phiên đầu. Checkpoint lớn và dataset không đưa vào repo. Notebook nguồn trong `code/` có các bước chạy lại từ đầu; thí nghiệm thật trên Colab có thêm bước khôi phục T02R sau khi phiên đầu mất checkpoint.
