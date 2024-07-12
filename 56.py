import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 设置文件夹路径和文件名
img_folder = 'CelebA/Img/img_align_celeba'
attr_file = 'CelebA/Anno/identity_CelebA.txt'

# 预定义训练集和测试集图像数
train_limit = 500
test_limit = 100

# 初始化图像和标签列表
images = []
labels = []

# 加载图像和标签数据
with open(attr_file, 'r') as f:
    lines = f.readlines()
    for line in lines[:train_limit + test_limit]:
        image_file = line.split()[0]  # 第一列为图像文件名
        label = int(line.split()[1])  # 第二列为标签信息，转换为整数类型
        image_path = os.path.join(img_folder, image_file)
        images.append(image_path)
        labels.append(label)  # 添加标签信息

# 将图像数据转换为 numpy 数组
labels = np.array(labels)
images = np.array(images)

# 使用PCA进行降维
num_components = 150  # 设定降维后的维度
pca = PCA(n_components=num_components, random_state=42)

# 将图像加载并展平为一维向量
images_flat = []
for img_path in images:
    img = Image.open(img_path)
    img = img.resize((64, 64))  # 调整图像大小为64x64（可根据需求调整）
    img_flat = np.array(img).flatten()  # 展平为一维数组
    images_flat.append(img_flat)

images_flat = np.array(images_flat)

# 进行PCA降维
images_pca = pca.fit_transform(images_flat)

# 使用KMeans进行聚类
num_clusters = len(np.unique(labels))  # 类别数
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(images_pca)

# 可视化聚类结果
plt.figure(figsize=(12, 6))

# 绘制PCA降维后的数据分布
plt.subplot(1, 2, 1)
plt.scatter(images_pca[:, 0], images_pca[:, 1], c=labels, cmap='tab10', alpha=0.5)
plt.title('PCA Components Scatter Plot')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Identity Labels')

# 绘制KMeans聚类结果
plt.subplot(1, 2, 2)
plt.scatter(images_pca[:, 0], images_pca[:, 1], c=cluster_labels, cmap='tab10', alpha=0.5)
plt.title('KMeans Clustering Scatter Plot')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster Labels')

plt.tight_layout()
plt.show()

# 输出聚类结果
print("KMeans 聚类结果:", cluster_labels)

# 可选：进行性能评估，比如轮廓系数
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(images_pca, cluster_labels)
print(f'平均轮廓系数: {silhouette_avg:.2f}')
