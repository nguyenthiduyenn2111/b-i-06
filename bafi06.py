import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# Hàm đọc và chuyển đổi ảnh thành RGB
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Hàm phân cụm ảnh bằng K-means
def apply_kmeans(image, n_clusters):
    pixels = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(pixels)
    labels = kmeans.labels_
    return labels.reshape(image.shape[:2])

# Hàm phân cụm ảnh bằng Fuzzy C-means
def apply_fcm(image, n_clusters):
    pixels = image.reshape((-1, 3)).T  # Chuyển vị để phù hợp với FCM
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        pixels, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None
    )
    labels = np.argmax(u, axis=0)
    return labels.reshape(image.shape[:2])

# Hàm hiển thị ảnh gốc và ảnh phân cụm
def display_results(images, titles, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.ravel()  # Chuyển đổi mảng 2D thành 1D để dễ duyệt
    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img, cmap='viridis' if 'Phân cụm' in title else None)
        axes[i].set_title(title)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

# Danh sách đường dẫn ảnh
image_files = ['images/bản đồ 1.jpg', 'images/bản đồ 1.jpg']
n_clusters = 2  # Số cụm cần phân

# Tạo danh sách để lưu kết quả hiển thị
all_images = []
all_titles = []

# Lặp qua từng ảnh để xử lý
for idx, file in enumerate(image_files):
    try:
        # Đọc và xử lý ảnh
        image = load_image(file)

        # Phân cụm bằng K-means và FCM
        kmeans_result = apply_kmeans(image, n_clusters)
        fcm_result = apply_fcm(image, n_clusters)

        # Thêm kết quả vào danh sách
        all_images.extend([image, kmeans_result, fcm_result])
        all_titles.extend([
            f"Ảnh gốc {idx + 1}",
            f"Phân cụm K-means {idx + 1}",
            f"Phân cụm Fuzzy C-means {idx + 1}"
        ])

    except ValueError as e:
        print(e)

# Hiển thị kết quả
display_results(all_images, all_titles, rows=len(image_files), cols=3)
