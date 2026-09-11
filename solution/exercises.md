# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature thấp (0.0–0.5), phản hồi có xu hướng ổn định, lặp lại và gần với thông tin chắc chắn; khi temperature tăng lên 1.0–1.5, câu trả lời trở nên đa dạng hơn, sáng tạo hơn nhưng cũng dễ sai lệch hoặc mất tính nhất quán. Nói ngắn gọn, temperature là mức độ “ngẫu nhiên” của model: thấp thì an toàn và lặp lại, cao thì sáng tạo và bất định hơn.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ chọn khoảng 0.2–0.5. Với chatbot hỗ trợ khách hàng, mục tiêu quan trọng nhất là tính chính xác, nhất quán và an toàn thông tin, nên temperature thấp giúp giảm nguy cơ nói sai, mơ hồ hoặc tạo ra câu trả lời “đẹp” nhưng không đáng tin.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Với 10.000 người dùng × 3 lượt/người/ngày = 30.000 gọi/ngày. Mỗi lượt 350 token đầu ra, nên gpt-4o tiêu tốn khoảng 350 / 1000 × 0.010 = 0.0035 USD mỗi lượt, còn gpt-4o-mini là 350 / 1000 × 0.0006 = 0.00021 USD mỗi lượt. Vậy gpt-4o đắt hơn khoảng 0.0035 / 0.00021 ≈ 16,7 lần, tức khoảng 17x. GPT-4o đáng dùng khi cần trả lời chất lượng cao trong các trường hợp như phân tích văn bản, hỗ trợ ra quyết định chuyên môn hoặc soạn email doanh nghiệp. Còn mini nên dùng cho chatbot hỗ trợ khách hàng nhanh, tóm tắt đơn giản, hoặc nhắn tin tự động khi cần tiết kiệm chi phí và phản hồi ngay.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Phản hồi của phiên bản giáo viên ngắn, dễ hiểu, dùng từ vựng gần gũi với trẻ em và có ví dụ như “blockchain giống như sổ nhật ký chia sẻ”. Còn phiên bản chuyên gia dài hơn, dùng thuật ngữ như “distributed ledger”, “consensus”, “immutability” và giải thích theo khung tài chính kỹ thuật. System prompt không chỉ thay đổi độ dài mà còn định hình cách suy nghĩ, mức độ chi tiết và góc nhìn của model. Vì vậy, cùng một câu hỏi nhưng khác persona sẽ cho ra cách diễn giải rất khác nhau dù cùng dựa trên kiến thức nền tương tự.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Với một đoạn văn tiếng Việt khoảng 100 từ, `count_tokens` thường cho ra số token lớn hơn một chút so với 100 / 0.75 ≈ 133 token, vì tokenizer tách theo đơn vị ngữ nghĩa và byte/Unicode chứ không chỉ theo khoảng trắng. Chênh lệch thường nằm trong khoảng 20–40% tùy đoạn văn, vì tiếng Việt có nhiều ký tự dấu và từ ghép, đồng thời tokenizer OpenAI phân tách theo mẫu ngôn ngữ khác hơn tiếng Anh, nên cùng 100 từ nhưng tiếng Việt có thể cần nhiều token hơn để biểu diễn cùng ý nghĩa. Đây là lý do vì sao tiếng Việt thường tốn nhiều token hơn tiếng Anh ở cùng độ dài.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất khi người dùng đang cần phản hồi ngay lập tức, ví dụ chat bot hỗ trợ khách hàng, trợ lý học tập, hoặc ứng dụng tổng hợp dài như mô tả/điều kiện. Nó tạo cảm giác nhanh, cho phép người dùng thấy tiến độ và tương tác sớm hơn. Non-streaming phù hợp hơn khi công việc cần tính chính xác và ổn định hơn, ví dụ generate một báo cáo ngắn, định dạng JSON, hoặc xử lý bài dài cần chờ đến khi toàn bộ kết quả được sinh xong trước khi hiển thị.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff giúp phân tán tải theo thời gian bằng cách tăng khoảng chờ sau mỗi lần thất bại, ví dụ 0.1s, 0.2s, 0.4s, 0.8s… Điều này làm giảm áp lực lên server khi quá tải và tránh tình trạng bầy đàn cùng gửi lại request cùng lúc. Nếu hàng nghìn client retry với cùng một delay cố định, họ sẽ đồng loạt kích hoạt lại trong cùng một khoảng thời gian, dẫn đến “thundering herd” — tức một đợt burst mới nặng hơn, khiến hệ thống càng quá tải, mất ổn định và dễ bị sập thêm.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Tôi chọn persona là “trợ lý học tập ngắn gọn bằng tiếng Việt”. System prompt mẫu: "Bạn là trợ lý học tập thân thiện, trả lời ngắn gọn bằng tiếng Việt, tập trung vào kiến thức cốt lõi, giải thích dễ hiểu và luôn giữ thái độ hỗ trợ, lịch sự." Từ "trả lời ngắn gọn" giúp tránh câu trả lời dài lan man và giữ tính thực dụng; từ "bằng tiếng Việt" đảm bảo phản hồi phù hợp với người dùng mục tiêu và đồng bộ ngữ cảnh, tránh lẫn tiếng Anh hoặc không phù hợp với đối tượng.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất của trợ lý là không có bộ nhớ dài hạn và history chỉ giữ tối đa 3 lượt gần nhất, nên nó dễ quên ngữ cảnh từ các cuộc trò chuyện trước đó. Một cải thiện cụ thể là lưu lịch sử dưới dạng summary + recent messages trong một cơ sở dữ liệu hoặc file session, rồi mỗi khi bắt đầu một lượt mới, tạo một “summary” ngắn từ hội thoại cũ và gửi cùng với context hiện tại. Cách này giúp trợ lý nhớ ý chính mà không làm tăng kích thước message quá lớn.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
