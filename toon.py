import cv2
import numpy as np
from pyparsing import col

def color_subtraction(image, div=4):
    # 画像処理100本ノック
    # https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/questions/question_01_10#q6-%E6%B8%9B%E8%89%B2-color-subtraction
    th1 = 256/div
    th2 = 256/(div-1)
    result = np.clip(image // th1 * th2 , 0, 255).astype(np.uint8)
    cv2.imwrite("gensyoku.png", result)
#    return img // th * th
#    result = img // th * th + th // 2
    return result

def blur(image, k=5):
    result = cv2.GaussianBlur(image, (k,k), 0)
    cv2.imwrite("blur.png", result)    
    return result

def canny(image):
    # cv2.Canny(): Canny法によるエッジ検出の自動化
    # https://qiita.com/kotai2003/items/662c33c15915f2a8517e
    sigma = 0.33
    med = np.median(image)
    th1 = int(max(0, (1-sigma)*med))
    th2 = int(max(255, (1+sigma)*med))
    result = cv2.Canny(image, th1, th2)
    return result

def bold(image, k=3):
    kernel = np.ones((k,k),np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

def otsu(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)[1]


def main():
    filename = "macho.jpg"
    image = cv2.imread(filename)

    gensyoku = color_subtraction(image)

    gray = cv2.cvtColor(gensyoku, cv2.COLOR_BGR2GRAY)
    img_blur = blur(image)
    gray_blur = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)


    edge = canny(gray_blur)

#    dil = dilation(edge)
#    edge2 = canny(edge)

    img_otsu = otsu(gray)
    img_otsu_edge = canny(img_otsu)
    img_otsu_edge_bold = bold(img_otsu_edge)
    img_otsu_edge_bold_3ch = cv2.merge((img_otsu_edge, img_otsu_edge, img_otsu_edge))

    img_final = np.where(img_otsu_edge_bold_3ch==(255,255,255), 255-img_otsu_edge_bold_3ch, gensyoku)

    cv2.imshow("image", image)
#    cv2.imshow("gensyoku", gensyoku)
#    cv2.imshow("edge", edge)
#    cv2.imshow("dil", dil)    
#    cv2.imshow("edge2", edge2)
#    cv2.imshow("otsu", img_otsu)
#    cv2.imshow("img_otsu_edge", img_otsu_edge)
#    cv2.imshow("img_otsu_edge_bold", img_otsu_edge_bold)
    cv2.imshow("img_final", img_final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__=="__main__":
    main()