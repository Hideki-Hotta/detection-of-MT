import glob
import re
import cv2
import os
import shutil


VIDEOPATH = "media/video/video.avi"
IMAGEPATH = "media/image/"
TEMPLATEPATH = "template_2.png"
RESULTPATH = "result_2.png"


def save_frames(video_path, image_dir):
    """
    動画からフレームの画像を抽出
    """
    cap = cv2.VideoCapture(video_path)
    print("VideoCapture:",cap.isOpened())
    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    print("FrameNumber:",cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = 0
    # while True:
    #     ret, frame = cap.read()
    #     if ret:
    #         cv2.imwrite("{}original/frame_{}.{}".format(IMAGEPATH, n, "png"), frame)
    #         n += 1
    #     else:
    #         return
    while n < 10:
        ret, frame = cap.read()
        cv2.imwrite("{}1_original/frame_{}.{}".format(IMAGEPATH, n, "png"), frame)
        n += 1



def do_grayscale(image_path):
    """
    画像をグレースケール化
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    save_image(image_path, "2_gray", gray)


def do_binarization(image_path):
    """
    画像を2値化
    """
    img = cv2.imread(image_path)
    ret, img_thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    save_image(image_path, "3_binary", img_thresh)


def do_backgroundsub():
    """
    背景差分を行う
    """
    img_list = glob.glob(IMAGEPATH + "3_binary/frame*.png")
    num = lambda val: int(re.sub("\D","",val))
    sorted(img_list,key=(num))
    source = img_list[0]
    for path in img_list:
        diff = cv2.absdiff(cv2.imread(source),cv2.imread(path))
        source = path
        save_image(path, "4_bgsub", diff)


def do_template_matching():
    """
    テンプレート画像とフレーム画像でテンプレートマッチングを行う
    """
    template_img = cv2.imread(IMAGEPATH + "3_binary/" + TEMPLATEPATH)
    img_list = glob.glob(IMAGEPATH + "4_bgsub/frame*.png")
    num = lambda val: int(re.sub("\D","",val))
    sorted(img_list,key=(num))
    location_list = []
    for path in img_list:
        result = cv2.matchTemplate(cv2.imread(path), template_img, cv2.TM_CCOEFF)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
        location_list.append(maxLoc)
    return location_list

def draw_rectangle(location_list):
    """
    マッチング結果を画像に描画する
    """
    source = cv2.imread(IMAGEPATH + "1_original/frame_0.png")
    cv2.imwrite(IMAGEPATH + RESULTPATH,source)
    source = cv2.imread(IMAGEPATH + RESULTPATH)
    for loc in location_list:
        lx, ly, rx, ry = loc[0] - 20, loc[1] - 80, loc[0] + 20, loc[1] + 80
        img = cv2.rectangle(source, (lx, ly), (rx, ry), (0, 255, 0), 3)
        cv2.imwrite(IMAGEPATH + RESULTPATH,img)

def save_image(img_path, dir, img):
    """
    画像を保存する
    img_path : 画像のパス
    dir : ディレクトリ名
    img : 画像データ
    """
    file_name = img_path.replace("\\","/").split(".")[0].split("/")[-1]
    cv2.imwrite("{}{}/{}.{}".format(IMAGEPATH, dir, file_name,"png"), img)

def remove_image(img_path,dir):
    """
    画像を削除する
    """
    shutil.rmtree(img_path + dir)
    os.mkdir(img_path + dir)


if __name__=="__main__":
    # ①動画をフレームごとに分割
    print("Now saving flames...")
    remove_image(IMAGEPATH,"1_original")
    save_frames(VIDEOPATH,IMAGEPATH)
    # ②テンプレート画像とフレーム画像をグレースケール化
    print("Now grayscaling...")
    remove_image(IMAGEPATH,"2_gray")
    do_grayscale(IMAGEPATH + TEMPLATEPATH)
    for path in glob.glob(IMAGEPATH + "1_original/*.png"):
        do_grayscale(path)
    # ③テンプレート画像とフレーム画像の2値化
    print("Now binaring...")
    remove_image(IMAGEPATH,"3_binary")
    for path in glob.glob(IMAGEPATH + "2_gray/*.png"):
        do_binarization(path)
    # ④背景差分を行う
    print("Now doing backgroudsub...")
    remove_image(IMAGEPATH,"4_bgsub")
    do_backgroundsub()
    # ⑤テンプレートマッチングを行う
    print("Now template matching...")
    location_list = do_template_matching()
    # ⑥マッチングした座標を投影
    print("Now drawing rectangle...")
    draw_rectangle(location_list)
