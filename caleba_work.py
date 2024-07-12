# 加载相关库
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


# K均值（K-Means）模型是一种无监督学习算法，用于将数据集分成预定的K个簇（cluster）。
# 这些簇的形成基于数据点之间的距离，
# 目标是使得同一簇内的数据点彼此更加相似，而不同簇之间的数据点尽可能不同。

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题


# 图像路径
image_folder = 'CelebA/Img/img_align_celeba'
image_label_file = 'CelebA/Anno/list_attr_celeba.txt'
categories = 5  # 假设我们根据某些特征分为5类（这个数字通常根据具体的问题和数据特征来选择，可以根据领域知识、数据分布以及实际需求来确定最合适的聚类数目。）
pca_components = 150  # PCA降维后的维度

# 加载图像标签
def load_labels(label_file):
    with open(label_file, 'r') as file:
        lines = file.readlines()
    labels = {}
    for line in lines[2:]:  # 跳过前两行,前两行是文字
        parts = line.strip().split()
        filename = parts[0] # 获取图片名称
        label = int(parts[1])  # 获取图片的标签
        # 检查标签是否在预期范围内
        if 0 <= label < categories:
            labels[filename] = label
    return labels

labels = load_labels(image_label_file)

# 选择每个类别的前500张图像作为训练样本和前100张图像作为测试样本
def select_images_per_category(labels, train_per_category, test_per_category):
    category_images = {i: [] for i in range(categories)}
    for filename, label in labels.items():
        if len(category_images[label]) < (train_per_category + test_per_category):
            category_images[label].append(filename)
    train_filenames = []
    test_filenames = []
    for label in range(categories):
        train_filenames.extend(category_images[label][:train_per_category])
        test_filenames.extend(category_images[label][train_per_category:train_per_category + test_per_category])
    return train_filenames, test_filenames

train_filenames, test_filenames = select_images_per_category(labels, 500, 100)

# 加载图像数据并提取特征
def load_images(filenames):
    images = []
    for filename in filenames:
        img = Image.open(os.path.join(image_folder, filename)).convert('L')  # 转换为灰度图像
        img = img.resize((64, 64))  # 调整图像大小
        img = np.array(img).flatten()  # 展平图像
        images.append(img)
    return np.array(images)

# 处理训练数据
train_data = load_images(train_filenames)
scaler = StandardScaler()
scaled_train_data = scaler.fit_transform(train_data)
pca = PCA(n_components=pca_components)
reduced_train_data = pca.fit_transform(scaled_train_data)

# 训练 KMeans 模型
kmeans = KMeans(n_clusters=categories, random_state=42)
kmeans.fit(reduced_train_data)

# 处理测试数据
test_data = load_images(test_filenames)
scaled_test_data = scaler.transform(test_data)
reduced_test_data = pca.transform(scaled_test_data)
test_labels = kmeans.predict(reduced_test_data)

# 可视化测试结果
plt.figure(figsize=(10, 8))
for cluster in range(categories):
    cluster_data = reduced_test_data[test_labels == cluster]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'簇 {cluster}')
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='black', label='聚心')
plt.title('CelebA 数据集的 KMeans 聚类')
plt.xlabel('主成分-1')
plt.ylabel('主成分-2')
plt.legend()
plt.show()

# 分析测试结果
for cluster in range(categories):
    print(f'簇 {cluster} has {np.sum(test_labels == cluster)} images')


# 计算轮廓系数
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(reduced_test_data, test_labels)
print(f'平均轮廓得分为: {silhouette_avg}')

# 简要分析
# 簇 0 has 22 images
# 簇 1 has 11 images
# 簇 2 has 34 images
# 簇 3 has 25 images
# 簇 4 has 8 images
# 平均轮廓得分为: 0.060275893159860934
# 平均轮廓得分为约0.0603，这表明聚类结果中的图像组内紧密度比组间的分离度稍高，但整体上聚类效果并不是非常明显。