from os.path import exists
from cv2 import imread, VideoWriter, VideoWriter_fourcc, imread, resize

def make_video(images:list, out_path:str=None, fps:int=5, size:tuple=None, is_color:bool=True, format:str='XVID'):
    """
    Create a video from a list of images.

    @param      out_path    output video
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
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        if type(image) == str:
            if exists(image):
                image = imread(image)
            else:
               raise FileNotFoundError(image)
        
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(out_path, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid