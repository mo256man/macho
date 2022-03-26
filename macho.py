from enum import auto
import cv2
import numpy as np
from pyparsing import col

def color_subtraction(image, div=4):
    th1 = 256 / div
    th2 = 256 / (div-1)
    result = np.clip(image // th1 * th2 , 0, 255).astype(np.uint8)
    return result

def blur(image, k=5):
    result = cv2.GaussianBlur(image, (k,k), 0)
    return result

def auto_canny(image):
    # cv2.Canny(): Canny法によるエッジ検出の自動化
    # https://qiita.com/kotai2003/items/662c33c15915f2a8517e
    sigma = 0.33
    med = np.median(image)
    th1 = int(max(0, (1-sigma)*med))
    th2 = int(max(255, (1+sigma)*med))
    canny = cv2.Canny(image, th1, th2)
    canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    return canny

def dilate(image, k=3):
    kernel = np.ones((k,k),np.uint8)
    return cv2.dilate(image, kernel)

def erode(image, k=3):
    kernel = np.ones((k,k),np.uint8)
    return cv2.erode(image, kernel)

def darken(image, edge):
    result = 1 / 256.0 * image * edge   # float64
    return result.astype(np.uint8)      # uint8にして返す


def main():
    filename = "macho_origin.jpg"
    img = cv2.imread(filename)

    # ベース画像　平滑化して減色する
    img_blur = blur(img)
    img_blur_subtract = color_subtraction(img_blur)

    # エッジ画像　平滑化してエッジ検出し膨張させさらに平滑化する
    img_blur_canny = auto_canny(img_blur)
    img_blur_canny_dilate = dilate(img_blur_canny)
    img_blur_canny_dilate_blur = blur(img_blur_canny_dilate)

    result = darken(img_blur_subtract, 255-img_blur_canny_dilate_blur)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()