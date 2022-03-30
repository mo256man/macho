import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def blur(image, k=5):
    result = cv2.GaussianBlur(image, (k,k), 0)
    return result

def auto_canny(image):
    sigma = 0.33
    med = np.median(image)
    th1 = int(max(0, (1-sigma)*med))
    th2 = int(max(255, (1+sigma)*med))
    canny = cv2.Canny(image, th1, th2)
    return canny

def dilate(image, k=3):
    kernel = np.ones((k,k),np.uint8)
    return cv2.dilate(image, kernel)

def erode(image, k=3):
    kernel = np.ones((k,k),np.uint8)
    return cv2.erode(image, kernel)

def find_edge(image):
    img_blur = blur(image)
    img_blur_canny = auto_canny(img_blur)
    return img_blur_canny

def count_edge(image):
    h, w = image.shape[:2]
    Y, X = np.where(image==255)   # 白のインデックスのリスト[[y1, y2, ...], [x1, x2, ...]]
    return np.array([X, Y]).T     # 転置して[[x1,y1], [x2,y2], ...]にする



filename = "macho.jpg"
img = cv2.imread(filename)
h, w = img.shape[:2]

edge = find_edge(img)
pos = count_edge(edge)
print (f"画素数={h*w} -> エッジ数={len(pos)}")

clustering = DBSCAN(eps=5, min_samples=5).fit(pos)
unique_labels = np.unique(clustering.labels_)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal', adjustable='box')
for label in unique_labels:
    p = pos[np.where(clustering.labels_==label)]
    plt.scatter(p[:,0], h-p[:,1], label=label)
ax.legend()
plt.show()
