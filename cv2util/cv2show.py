import cv2 as cv, copy, numpy as np, os
from os.path import splitext

class cv2show:
    def __init__(self, img_format='png', display:str=None, resize:tuple=None, wait_time:int=1, separator_thick:int=0, separator_color:tuple=(255, 255, 255)):
        self.__concated_frame = None
        self.ext = img_format
        self.resize = resize
        self.wait_time = wait_time
        self.separator_thick = separator_thick
        self.separator_color = separator_color
        self.pause=False
        self.skipped=False
        if display:
            os.system(f'export DISPLAY={display}')
    
    def add(self, frame, text:str=None, grayscale:bool=False, binarize:bool=False, bin_thr:int=0, bin_max:int=255, font_scale:float=0.5, text_position:tuple=(10, 20), text_color:tuple=(255, 255, 255)):
        if type(frame) == str:
            frame = cv.imread(frame, cv.IMREAD_GRAYSCALE if grayscale else cv.IMREAD_COLOR)
        else:
            frame = copy.deepcopy(frame)

        # Check the integrity
        if len(frame.shape) == 2 and not grayscale:
            frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        if frame.dtype != np.uint8:
            frame = np.array(frame, dtype=np.uint8)

        # Apply style
        if self.resize:
            frame = cv.resize(frame, dsize=(self.resize[1], self.resize[0]))
        if binarize:
            frame = cv.threshold(frame, bin_thr, bin_max, cv.THRESH_BINARY)[1]
        if text:
            cv.putText(frame, text, text_position, cv.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=text_color, lineType=cv.FILLED)
        if self.separator_thick and self.__concated_frame is not None:
            # https://docs.opencv.org/3.0-last-rst/modules/core/doc/operations_on_arrays.html?highlight=copymakeborder#copymakeborder
            self.__concated_frame = cv.copyMakeBorder(self.__concated_frame, 0, 0, 0, self.separator_thick, cv.BORDER_CONSTANT, value=self.separator_color)
       
        # Concatenate new frame
        if self.__concated_frame is None:
            self.__concated_frame = frame
            self.resize = frame.shape
        else:
            self.__concated_frame = cv.hconcat([self.__concated_frame, frame])
        return self
    
    def close(self):
        cv.destroyAllWindows()
        self.__concated_frame = None
    
    def show(self, window_title:str='OpenCV display', flush:bool=True, wait_key:int=None):
        if self.skipped:
            cv.destroyAllWindows()
            return

        if wait_key is not None:
            self.wait_time = wait_key
        
        cv.imshow(window_title, self.__concated_frame)
        # cv.namedWindow(window_title, cv.WINDOW_AUTOSIZE)
        cv.setWindowProperty(window_title, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        # cv.setWindowProperty(window_title, cv.WND_PROP_TOPMOST, 1)
        # cv.setWindowProperty(window_title, cv.WND_PROP_AUTOSIZE, cv.WINDOW_AUTOSIZE)
        # cv.setWindowProperty(window_title, cv.WND_PROP_AUTOSIZE, cv.WINDOW_NORMAL)
        # cv.setWindowProperty(window_title, cv.WND_PROP_ASPECT_RATIO, cv.WINDOW_KEEPRATIO)
        
        # Keyboard interaction
        key = cv.waitKey(0 if self.pause else self.wait_time)
        if key == 32:
            if self.pause:
                self.pause = False
            else:
                self.pause = True
        elif key == 27:
            self.skipped = True

        if flush:
            self.__concated_frame = None

    def save(self, path, flush=True):
        if self.__concated_frame is not None:
            cv.imwrite(splitext(path)[0] + '.' + self.ext, self.__concated_frame)
        else:
            print('Cannot save the result image because the buffer is empty.')
        if flush:
            self.__concated_frame = None
