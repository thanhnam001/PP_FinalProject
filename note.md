A. Tuần tự
- find energy:
    - chuyển ảnh RGB sang grey
    - edge detect: x-sobel + y-sobel
- seam ít quan trọng:
    - bảng quy hoạch động ma trận tính seam min
    - backtrack đường seam nhỏ nhất
- xóa seam min
- lặp đến khi đạt kích thước

B. Song song
    1. chuyển ảnh RGB sang grey -> done
    2. sobel
    3. quy hoạch động ra ma trận các seam
    4. tìm seam nhỏ nhất trên hàng cuối
    5. backtracking
    6. xóa seam
