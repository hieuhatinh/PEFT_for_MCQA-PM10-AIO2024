# Project M10 - PEFT (Parameter-Efficient Fine-Tuning) for Multiple-choice Question Answering - AIO2024

**Parameter-Efficient Fine-Tuning (PEFT)** là một nhóm các kỹ thuật nhằm giảm số lượng tham số cần cập nhật trong quá trình fine-tune, giữ nguyên phần lớn trọng số gốc của mô hình, từ đó giúp tăng hiệu quả huấn luyện và khả năng mở rộng.

Mục tiêu chính của PEFT:
- Tận dụng sức mạnh của mô hình tiền huấn luyện (pre-trained).<br>
- Giảm số lượng tham số cần cập nhật trong quá trình tinh chỉnh.<br>
- Dễ dàng chia sẻ, lưu trữ và tái sử dụng các mô hình đã tinh chỉnh.<br>

**1. Subset Fine-tuning**
- là một phương pháp tinh chỉnh hiệu quả tham số bằng cách chỉ tinh chỉnh một phần (subset) các tầng của mô hình, giữ nguyên (freeze) các tầng còn lại. Cơ chế hoạt động: Chỉ top-K tầng cuối cùng (thường là những tầng gần output) được fine-tune, các tầng trước đó được giữ nguyên. <br>

***Ưu điểm:***<br>
- Giảm chi phí huấn luyện vì số lượng tham số cập nhật ít hơn.<br>
- Dễ triển khai trên hầu hết các kiến trúc Transformer hiện có.<br>

***Hạn chế:***<br>
- Không tận dụng toàn bộ cấu trúc mô hình để thích nghi với dữ liệu mới.<br>
- Dễ bị overfit nếu fine-tune quá ít tầng hoặc underfit nếu chọn tầng không phù hợp.<br>

**2. Adapter-tuning**<br>
- là các module nhỏ được chèn vào giữa các tầng của mô hình gốc. Mô hình chính được giữ nguyên, chỉ các module Adapter là được huấn luyện.<br>
Kiến trúc Adapter phổ biến: Down-projection → Non-linearity → Up-projection (thường với
 bottleneck rank nhỏ, ví dụ: 16, 32)<br>

 ***Ưu điểm:***<br>
 - Có thể dễ dàng huấn luyện nhiều task khác nhau bằng cách thay Adapter.<br>
 - Thích hợp cho multi-task learning hoặc deployment nhiều mô hình nhẹ.<br>
 - Cập nhật ít tham số nhưng vẫn đạt hiệu quả cao.<br>

 ***Hạn chế:***<br>
 - Cần chèn các module vào mô hình, đòi hỏi can thiệp code.<br>
 - Tăng độ trễ nhẹ trong quá trình inference.<br>

**3. Prefix Tuning**<br>
- là kỹ thuật tinh chỉnh bằng cách thêm các vector học được (prefix vectors) vào
 phần đầu của chuỗi đầu vào trong mỗi layer, đồng thời giữ nguyên toàn bộ mô hình gốc.<br>
Cách hoạt động:<br>
- Huấn luyện các prefix embedding, thường biểu diễn attention key/value bổ sung.<br>
- Không can thiệp trực tiếp vào tham số gốc của mô hình.<br>

 ***Ưu điểm:***<br>
 - Số lượng tham số cần huấn luyện cực kỳ nhỏ (thường <1%).<br>
 - Dễ mở rộng sang nhiều tác vụ khác nhau.<br>

  ***Hạn chế:***<br>
  - Phụ thuộc vào khả năng của mô hình trong việc xử lý prefix (không phải mô hình nào cũng hỗ trợ).<br>
  - Cần số lượng prefix tương ứng với mỗi tầng.<br>

**4. Low-rank Adaptation (LoRA):** 
- là kỹ thuật tinh chỉnh bằng cách chèn ma trận low-rank để mô phỏng sự thay đổi tham số trong các tầng Attention.<br>
Cách hoạt động:<br>
- Các ma trận trọng số W được đóng băng.<br>
- Thay vào đó, huấn luyện hai ma trận A và B có rank thấp sao cho: W′ = W +W, W = AB<br>

 ***Ưu điểm:***<br>
 - Tối ưu về bộ nhớ: số tham số học được giảm mạnh (rank thấp).<br>
 - Hiệu quả tương đương hoặc tốt hơn full fine-tuning trong nhiều task.<br>
 - Không cần can thiệp quá sâu vào mô hình, dễ áp dụng.<br>

 ***Hạn chế:***<br>
 - Cần chọn rank hợp lý (quá thấp → thiếu năng lực học, quá cao → tốn tài nguyên).<br>
 - Dễ overfit nếu áp dụng không đúng cấu hình.<br>

**5. QLoRA:**
![Minh họa LoRA và QLoRA](/readme_img/illustration_lora-qlora.png 'AIO2024')
- là viết tắt của Quantized Low-Rank Adapter, là một phương pháp fine-tuning cực kỳ tiết kiệm tài nguyên, cho phép huấn luyện các mô hình ngôn ngữ cực lớn (tới 65B tham số) ngay cả trên GPU 24 hoặc 48 GB (consumer GPUs).<br>

Mục tiêu chính của QLoRA:<br>
- Kết hợp sức mạnh của LoRA với mô hình được lượng tử hóa (quantized).<br>
- Giảm tối đa dung lượng bộ nhớ và chi phí tính toán khi fine-tune.<br>
- Giữ hiệu năng tương đương hoặc tốt hơn so với full fine-tuning trong nhiều tác vụ.<br>

QLoRA kết hợp 3 ý tưởng chính:<br>
- 4-bit Quantization của mô hình gốc
Trước tiên, mô hình gốc (pre-trained model) được lượng tử hóa về 4-bit bằng kỹ thuật NF4
 (Normalized Float 4-bit) – một định dạng mới giữ lại tốt hơn phân phối gốc của trọn số. Điều này giúp giảm đáng kể bộ nhớ RAM/GPU, cho phép chạy mô hình lớn hơn rất nhiều.
- Low-Rank Adapter
Mô hình gốc được giữ nguyên (sau khi lượng tử hóa).
 Các ma trận Low-Rank (A × B) như trong LoRA sẽ được huấn luyện thêm, nhưng vẫn ở độ
 chính xác float32 hoặc bfloat16.
 Như vậy, chỉ có rất ít tham số cần học (low-rank), và không cần cập nhật mô hình gốc.
- Double Quantization
Tăng cường nén bằng cách tiếp tục lượng tử hóa các giá trị lượng tử hóa – giúp giảm thêm nhu cầu lưu trữ mà không làm mất thông tin.

Dưới đây là 2 phần thực hiện trong việc áp dụng PEFT for MCQA: 
- Preprocessing Stage
- Training Stage
![2 stages](/readme_img/2-stages.png 'AIO2024')