import sys, os
from os.path import join
sys.path.append(join(os.getcwd()))
from cv2util.cv2show import cv2show
# from cv2util import cv2show
# import cv2show
# from cv2show import __version__


# def test_version():
    # assert __version__ == '0.1.0'

import cv2 as cv

if __name__ == '__main__':
    show = cv2show(resize=(240, 320), separator_thick=1, wait_time=100)
    from glob import glob
    x1_paths = sorted(glob('tests/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/Train001/*.tif'))
    x2_paths = sorted(glob('tests/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/Train001-x2/*.png'))
    flow_paths = sorted(glob('tests/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/optflow/*.png'))
    contour_paths = sorted(glob('tests/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/contour/*.png'))
    bg_path = sorted('tests/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/background/train.png')
    # for path in paths:
    #     cv.imshow('show', cv.imread(path))
    #     cv.waitKey(1)
        
    for x1_path, x2_path, flow_path, contour_path in zip(x1_paths[1:], x2_paths[1:], flow_paths, contour_paths[1:]):
        x1 = cv.imread(x1_path)
        x2 = cv.imread(x2_path, cv.IMREAD_GRAYSCALE)
        flow = cv.imread(flow_path, cv.IMREAD_REDUCED_COLOR_4)

        show.add(x1, 'Original frame')
        show.add(x2, 'X2 frame', text_color=(255, 0, 255))
        show.add(flow)
        show.add(contour_path)
        show.show(flush=False)
        show.save('results/' + os.path.basename(x1_path))
