# =============================================================================
"""
Code Information:
    Date: 04/05/2019
	Programmer: John A. Betancourt G.
	Mail: john.betancourt93@gmail.com
    Web: www.linkedin.com/in/jhon-alberto-betancourt-gonzalez-345557129

Description: Project 4 - Udacity - self driving cars Nanodegree
    (Deep Learning) Driving Behavioral Cloning

Tested on: 
    python 2.7 (3.X should work)
    OpenCV 3.0.0 (3.X or 4.X should work)
    UBUNTU 16.04
"""

# =============================================================================
# LIBRARIES AND DEPENDENCIES - LIBRARIES AND DEPENDENCIES - LIBRARIES AND DEPEN
# =============================================================================
#importing useful packages
import numpy as np
import cv2
import csv
import os

import matplotlib.pyplot as plt

# =============================================================================
def print_list_text(img_src, str_list, origin=(0, 0), color=(0, 255, 255), 
    thickness=2, fontScale=0.45,  y_space=20):

    """  prints text list in cool way
    Args:
        img_src: `cv2.math` input image to draw text
        str_list: `list` list with text for each row
        origin: `tuple` (X, Y) coordinates to start drawings text vertically
        color: `tuple` (R, G, B) color values of text to print
        thickness: `int` thickness of text to print
        fontScale: `float` font scale of text to print
        y_space: `int` [pix] vertical space between lines
    Returns:
        img_src: `cv2.math` input image with text drawn
    """

    for idx, strprint in enumerate(str_list):
        cv2.putText(img = img_src,
                    text = strprint,
                    org = (origin[0], origin[1] + (y_space * idx)),
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = fontScale, 
                    color = (0, 0, 0), 
                    thickness = thickness+3, 
                    lineType = cv2.LINE_AA)
        cv2.putText(img = img_src,
                    text = strprint,
                    org = (origin[0], origin[1] + (y_space * idx)),
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = fontScale, 
                    color = color, 
                    thickness = thickness, 
                    lineType = cv2.LINE_AA)

    return img_src

def load_dataset(data_path):
    
    """ load dataset from path 
    Args:
        data_path: `string` path where images files are located
    Returns: 
        images: `list` of cv2.math of images 
        measurements: `list` of stering measuments
    """

    lines = []
    with open(data_path) as csvfile:
        reader = csv.reader(csvfile)
        for idx, line in enumerate(reader):
            if idx:
                lines.append(line)

    folder_path = os.path.dirname(data_path)

    images_c = np.asarray([cv2.imread(os.path.join(folder_path, lines[idx][0])) for idx in range(len(lines))])
    images_l = np.asarray([cv2.imread(os.path.join(folder_path, lines[idx][1])) for idx in range(len(lines))])
    images_r = np.asarray([cv2.imread(os.path.join(folder_path, lines[idx][2])) for idx in range(len(lines))])
    measurements = np.asarray([float(lines[idx][3]) for idx in range(len(lines))])

    return images_c, images_l, images_r, measurements

def create_video(images, path, file_name, fps=15., video_size=None):

    """ Create video from images list 
    Args:
        images" `list` of cv2.math images 
        path" `string` path to save videos
        file_name" `string` video file name to save
        fps" `float` desired video frame rate
        video_size" `tuple` desired video size (widht, height)
    Returns: 
    """

    # video size
    if video_size is None:
        video_size = (images.shape[0], images.shape[1])

    # Define the codec and format and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec H264, format MP4
    video_name = os.path.join(path, file_name) # File name and format
    video_out = cv2.VideoWriter(video_name, fourcc, fps, video_size) # Video recorder variable

    # Write frame to video
    for img in images:
        video_out.write(cv2.resize(img, video_size))

    # Release video variable memory
    video_out.release()

def reproduce_dataset(images, fps=15., loop=True, sc_fc=2., up_limit=0, down_limit=0):

    """ show dataset 
    Args:
        images" `list` of cv2.math images 
        fps" `float` desired video frame rate
        loop" `boolean` enable/disable looping
        sc_fc" `float` scaling factor to show video
    Returns: 
    """

    down_limit = images.shape[1]-down_limit

    reproduce = True
    while True:
        idx = 0
        while idx in range(len(images)):

            if reproduce:

                img = images[idx]

                if up_limit or down_limit:
                    img2 = cv2.rectangle(img.copy(),(0,0),(img.shape[1], up_limit),(0,0,0), -1)
                    img = cv2.line(img,(0,up_limit),(img.shape[1],up_limit),(0,0,255),2)
                    img2 = cv2.rectangle(img2,(0,down_limit),(img.shape[1],img.shape[0]),(0,0,0), -1)
                    img = cv2.line(img,(0,down_limit),(img.shape[1],down_limit),(0,0,255),2)
                    img = cv2.addWeighted(src2 = img2, src1 = img, 
                        alpha = 0.2, beta = 1, gamma = 0)
                img = cv2.resize(img, 
                    (int(images.shape[0]*sc_fc), int(images.shape[1]*sc_fc)))

                str_list = ("A: stop/reproduce", "Q: Quit", "{}%".format(round(idx*100./len(images), 2)))
                print_list_text(
                    img, str_list, origin=(10, 15), color=(0, 255, 255), 
                    thickness=1, fontScale=0.50,  y_space=20)
                cv2.imshow("data_visualization", img)

                # Increment index
                idx =-1 if loop and idx == len(images) -1 else (idx+1 if reproduce else idx)
            
            user_inpu = cv2.waitKey(int(1000/fps)) & 0xFF
            if user_inpu == ord('q') or  user_inpu == ord('Q'): break
            if user_inpu == ord('a') or  user_inpu == ord('A'): reproduce = not reproduce

        if user_inpu == ord('q') or user_inpu == ord('Q'): break
        if idx >= len(images) -1: break
    
    # Destroy GUI/User interface window
    cv2.destroyAllWindows()

def plot_data_distribution(y_data, scfc=30):

    plt.figure(figsize=(15,5))
    plt.hist(y_data, density=False, bins= [bin/scfc for bin in range(-scfc, scfc, 1)], facecolor='g', alpha=0.75)
    plt.xlabel('Steering Angles')
    plt.ylabel('Distirbutions')
    plt.title('Distirbutions of Steering Angles')
    plt.grid(True)
    plt.show()



# =============================================================================
