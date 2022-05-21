import cv2 as cv, copy, numpy as np, os
from glob import glob
import re
from os.path import join, basename

class Preview:
    def __init__(self, display, resize=None, wait_time=1):
        self.frames = None
        self.resize = resize
        self.wait_time = wait_time
        os.system(f'export DISPLAY={display}')
    
    def append(self, frame, text=None, channels=3, binarize=False, font_scale=0.5, text_color=(255, 255, 255), border_size=1, border_color=(255, 255, 255)):
        frame = copy.deepcopy(frame)

        # Check the integrity
        if frame.dtype != np.uint8:
            frame = np.array(frame, dtype=np.uint8)

        if len(frame.shape) == 2 and channels == 3:
            frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        if frame.dtype != 'uint8':
            frame = np.array(frame, dtype=np.uint8)

        if self.resize is not None:
            frame = cv.resize(frame, dsize=(self.resize[1], self.resize[0]))
        
        if binarize:
            frame = cv.threshold(frame, 0, 255, cv.THRESH_BINARY)[1]

        if not text is None:
            cv.putText(frame, text, (10, 20), cv.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=text_color, lineType=cv.FILLED)
        
        if border_size > 0:
            # https://docs.opencv.org/3.0-last-rst/modules/core/doc/operations_on_arrays.html?highlight=copymakeborder#copymakeborder
            frame = cv.copyMakeBorder(frame, 0, 0, 0, border_size, cv.BORDER_CONSTANT, value=border_color)

        if self.frames is None:
            self.frames = frame
        else:
            self.frames = cv.hconcat([self.frames, frame])
        
        return self
    
    def get_concat(self):
        return self.frames

    def close(self):
        cv.destroyAllWindows()
        self.frames = None
    
    def show(self, window_title, flush=True, wait_key:int=None):
        if wait_key is not None:
            self.wait_time = wait_key

        cv.imshow(window_title, self.frames)
        # cv.namedWindow(window_title, cv.WINDOW_AUTOSIZE)
        cv.setWindowProperty(window_title, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        # cv.setWindowProperty(window_title, cv.WND_PROP_TOPMOST, 1)
        # cv.setWindowProperty(window_title, cv.WND_PROP_AUTOSIZE, cv.WINDOW_AUTOSIZE)
        key = cv.waitKey(self.wait_time)
        if key == 32:
            if self.wait_time:
                self.wait_time = 0
            else:
                self.wait_time = wait_key if wait_key is not None else 1

        # cv.setWindowProperty(window_title, cv.WND_PROP_AUTOSIZE, cv.WINDOW_NORMAL)
        # cv.setWindowProperty(window_title, cv.WND_PROP_ASPECT_RATIO, cv.WINDOW_KEEPRATIO)
        if flush:
            self.frames = None

    def save(self, path, flush=True):
        cv.imwrite(path, self.frames)
        if flush:
            self.frames = None

def fuse(image0, image1, value):
    res = copy.deepcopy(image0)
    # res = cv.bitwise_and(image1, image1, mask=image0)  # https://076923.github.io/posts/Python-opencv-32/
    res[image1 == 0] = value
    return res

def make_video(images, outvid=None, fps=5, size=None,
               is_color=True, format="XVID"):
    """
    Create a video from a list of images.

    @param      outvid      output video
    @param      images      list of images to use in the video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

    The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
    By default, the video will have the size of the first image.
    It will resize every image to this size before adding them to the video.
    """
    from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
    import os
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid

def quantization(img, div=128):
    return cv.normalize(img // div * div + div // 2, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

def contrast(img, threshold, alpha):
    # b, g, r = cv.split(img)
    # z = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
    # s = cv.merge((z, g, z))

    con = np.clip(img + ((img * 1.0) - threshold) * alpha, 0, 255).astype(np.uint8)
    return con

def binarize(img, thr=0, value=1):
    return cv.threshold(img, thr, value, cv.THRESH_BINARY)[1]

def concat_frame(src_path, dtype, skip_each_first=False, reverse_sort=False):
    buffer = []
    file_paths = sorted(glob(src_path), reverse=reverse_sort)
    for path in file_paths:
        loaded = np.load(path, allow_pickle=True)[1 if skip_each_first else 0:]
        buffer.append(loaded)
    
    # Numpization
    bufferd_npy = np.concatenate(([i for i in buffer]), axis=0).astype(dtype)
    return bufferd_npy



if __name__ == '__main__':
    import glob
    from os.path import join

    sources_dir = '/root/workspace/results/ucsd_ped2/output/output_09-26-15-53'
    sources_list = sorted(glob.glob(join(sources_dir, '*.png')))
    make_video(sources_list, 'ped2.avi', 30)
