import numpy as np
import cv2
import random
import math

# 可視化
import matplotlib as mpl
import matplotlib.pyplot as plt

# 正規化のためのクラス
from sklearn.preprocessing import StandardScaler

# k-means法に必要なものをインポート
from sklearn.cluster import KMeans


def make_cluster(x0, y0, r0, n):
    positions = []
    for _ in range(n):
        r = random.random()
        angle = 2*math.pi*random.random()
        x = x0 + int(r0*r*math.cos(angle))
        y = y0 + int(r0*r*math.sin(angle))
        positions.append((x,y))
    return positions                                # [(x1,y1), (x2,y2), (x3,y3), …]

def make_sample_data():
    pos1 = make_cluster(200, 100, 100, 100)
    pos2 = make_cluster(400, 300, 150, 500)
    pos3 = make_cluster(100, 300, 50, 100)
    pos4 = (500, 50)                                # 孤立した点
    return np.vstack([pos1, pos2, pos3, pos4])

def show_graph_plt(pos, size):
    w, h = size
    X = [p[0] for p in pos]     # [x1, x2, x3, …]
    Y = [p[1] for p in pos]     # [y1, y2, y3, …]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X, Y)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    plt.show()

def make_graph_cv(positions, size, labels=None):
    colors = [(0,0,0), (0,0,255), (0,255,0), (255,0,0)]     # クラスタ数3の色定義　必要に応じて変える
    width, height = size
    if labels is None:                                      # ラベルなしの場合は
        labels = [-1 for x in range(len(positions))]        # 全要素が-1のラベルを作る（あとで+1する）
    image = np.full((height,width,3), (255,255,255), np.uint8)
    for pos, label in zip(positions, labels):
        color = colors[label+1]                             # ラベルなし→0番目の色、ラベルあり→は0,1,2が1,2,3番の色
        image = cv2.drawMarker(image, pos, color, cv2.MARKER_CROSS, 10)
        # image[y][x] = color
    return image


if __name__ == "__main__":
    K = 3                               # クラスタ数
    width, height = 640, 480            # グラフのサイズ
    positions = make_sample_data()

    # matplotlib.pyplot で表示するならこちら
    # show_graph_plt(pos, size)

    # cv2.kmeans()
    input_data = np.array(positions, np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    compactness, labels, centers = cv2.kmeans(input_data, K, bestLabels=None, criteria=criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)
    labels = [x[0] for x in labels]
     

    # OpenCVで作図
    img_corigin = make_graph_cv(positions, (width, height), None)

    # OpenCVでクラスター作図
    img_cluster = make_graph_cv(positions, (width, height), labels)
    for center in centers:
        center = center.astype(np.int32)                    # float32 -> int32
        cv2.circle(img_cluster, center, 5, (0,0,0), -1)     # クラスターの重心に印をつける


    cv2.imshow("origin", img_corigin)
    cv2.imshow("cluster", img_cluster)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
