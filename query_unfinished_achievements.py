"""
说明：
实现原理：录屏成就-->视频抽帧-->图片识别处理-->与全成就相减

这里面存在一些逻辑错误，比如一个成就达成30/60/90分别会给原石，那么这个算三个成就
但是由于easyocr识别不是很准确，所以采用了比较极端的数据清洗方式
导致不好去类比那些梯度成就是做过的，因此这里只是为了方便查找那些未完成成就

APP端没有做测试，但是应该也是能识别处理的
PC端需要注意自己电脑的分辨率是多少，我这里用的1920x1080
如果像素不相同，需要自行调整坐标轴，即left，right，top，bottom

图片处理默认使用GPU，如果没有可以将‘gpu=True’改为‘gpu=False’，使用CPU
"""

from PIL import Image
import easyocr
import cv2
import numpy as np
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import queue

# 重新封装线程池类
class ThreadPool_Executor(ThreadPoolExecutor):
    """
    重写线程池修改队列数
    """

    def __init__(self, max_workers=None, thread_name_prefix=''):
        super().__init__(max_workers, thread_name_prefix)
        # 队列大小为最大线程数的两倍
        self._work_queue = queue.Queue(self._max_workers * 2)

threadsPool = ThreadPool_Executor(max_workers=10)  # 定义线程数量

# 通过录屏原神成就抽帧处理
def frame_video(video_path,save_path):
    # 读取视频文件
    video = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not video.isOpened():
        print("无法打开视频文件")
        exit()

    # 获取视频的帧率
    fps = int(video.get(cv2.CAP_PROP_FPS))

    # 设置抽帧间隔
    frame_interval = fps

    frame_count = 0
    while True:
        # 读取下一帧
        ret, frame = video.read()

        # 如果读取失败，跳出循环
        if not ret:
            break

        # 判断是否需要保存当前帧
        if frame_count % frame_interval == 0:
            # 保存帧到指定路径
            cv2.imwrite(save_path.replace("xxx",str(frame_count)), frame)

        # 更新帧计数器
        frame_count += 1
    print("抽帧完成")

    # 释放视频资源
    video.release()
    cv2.destroyAllWindows()

# 处理并识别抽帧完成后的成就图片，这个时间比较长，可以开多线程处理
def ys_achieve(path) -> list:
    files = os.listdir(path)
    achieve_list = []
    task_list = []
    for file in files:
        # 打开图片
        image = Image.open(path+"\\"+file)

        # 获取原始宽高
        width, height = image.size
        print(f"原始宽度： {width}, 原始高度： {height}")

        # 设置截取区域的左上角和右下角坐标
        left = 800
        top = 150
        right = width - 100
        bottom = height - 100

        # 截取图片
        cropped_image = image.crop((left, top, right, bottom))

        # 展示截取后的图片
        # cropped_image.show()
        task = threadsPool.submit(_threads,cropped_image)
        achieve_list += task_list.append(task)
    return list(set(achieve_list))

def _threads(img_ele):
    img = cv2.cvtColor(np.array(img_ele), cv2.COLOR_RGB2BGR)

    # 识别截取后的图片内容
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)
    result = reader.readtext(img, detail=0)

    # 对列表元素清洗
    filtered_list = [item for item in result if "。" not in item and "达成" not in item]
    chinese_only_list = [item for item in filtered_list if all(char.isdigit() == False and char != '/' for char in item)]

# 原神截止到目前版本共计多少成就
def ys_all_achieve(path) -> list:
    # 读取Excel文件
    df = pd.read_excel(path, engine='openpyxl')

    # 获取第一列数据
    first_column = df.iloc[:, 0]

    return first_column

if __name__ == "__main__":
    video_path = r"D:\PR\视频剪辑\动漫游戏\20240804_105647.mp4" # 视频文件路径
    save_path = r"D:\PS\test\xxx.jpg" # 保存图片路径
    file_path = r"D:\PS\test" # 获取图片路径
    excel_path = r'./原神 成就.xlsx' # 成就文件路径

    frame_video(video_path,save_path)

    finished_achieve = ys_achieve(file_path)
    print(f"游戏内共计完成{len(finished_achieve)}个成就")
    print("----------------------------------------")
    print(finished_achieve)
    print("----------------------------------------")

    all_achieve = ys_all_achieve(excel_path)
    # 剩余的原神成就
    remain_achieve = list(set(all_achieve) - set(finished_achieve))
    print(f"游戏内未完成{len(remain_achieve)}个成就")
    print("----------------------------------------")
    print(remain_achieve)
    print("----------------------------------------")