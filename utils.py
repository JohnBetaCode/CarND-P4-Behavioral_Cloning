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
import glob
import cv2
import csv
import os

import matplotlib.pyplot as plt

import subprocess

# =============================================================================
def getClipboardData():
    p = subprocess.Popen(['xclip','-selection', 'clipboard', '-o'], stdout=subprocess.PIPE)
    retcode = p.wait()
    data = p.stdout.read()
    return data

def setClipboardData(data):
    p = subprocess.Popen(['xclip','-selection','clipboard'], stdin=subprocess.PIPE)
    p.stdin.write(data)
    p.stdin.close()
    retcode = p.wait()

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

def load_dataset(data_paths, csv_name="driving_log.csv"):
    
    """ load dataset from path 
    Args:
        data_paths: `list` of sting with paths to datasets
        csv_name: `string` name of csv files 
    Returns: 
        data" `list` of dictionaries with datasets information
    """

    data = []
    for idx_path, data_path in enumerate(data_paths):
        data_path = os.path.join(data_path, csv_name)
        with open(data_path) as csvfile:
            reader = csv.reader(csvfile)
            for idx, line in enumerate(reader):
                if idx:
                    data.append({
                        "img_c":line[0],
                        "img_l":line[1],
                        "img_r":line[2],
                        "steering": float(line[3]),
                        })
            print("From .... {}: {} samples".format(data_paths[idx_path][-30:], idx))

    return data

def create_video(cam_label, dst_path, src_path, file_name, fps=15., 
    video_size=(320, 160)):

    """ Create video from images list 
    Args:
        cam_label" `string` camera label to record video
        dst_path" `string` path to save videos
        src_path" `string` path  where data is stored
        file_name" `string` video file name to save
        fps" `float` desired video frame rate
        video_size" `tuple` desired video size (width, height)
    Returns: 
    """

    data = load_dataset([src_path], csv_name="driving_log.csv")

    # Define the codec and format and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec H264, format MP4
    video_name = os.path.join(dst_path, file_name) # File name and format
    video_out = cv2.VideoWriter(video_name, fourcc, fps, video_size) # Video recorder variable

    # Write frame to video
    for idx in range(len(data)):
        img = cv2.imread(os.path.join(dst_path, 'IMG', data[idx][cam_label]))
        video_out.write(cv2.resize(img, video_size))

    # Release video variable memory
    video_out.release()

def reproduce_dataset(fps=15., loop=True, sc_fc=2., up_limit=0, 
    down_limit=0, dataset_path="", CORRECTION=0.2):

    """ show dataset 
    Args:
        fps: `float` desired video frame rate
        loop: `boolean` enable/disable looping
        sc_fc: `float` scaling factor to show video
        up_limit: `int` superior limit to ignore image
        down_limit: `int` inferior limit to ignore image
        dataset_path: `string` path where dataset `data` is stored
        CORRECTION: `float` angle correction factor
    Returns: 
    """

    data = load_dataset([dataset_path], csv_name="driving_log.csv")

    cam_labels = ("img_l", "img_c", "img_r")
    reproduce = True
    while True:
        idx = 0
        while idx in range(len(data)):
            if reproduce:
                imgs = []
                for label in cam_labels:
                    img = cv2.imread(data[idx][label])
                    if up_limit or down_limit:
                        # Draw superior limit
                        img2 = cv2.rectangle(img.copy(),(0,0),(img.shape[1], up_limit),(0,0,0), -1)
                        img = cv2.line(img,(0,up_limit),(img.shape[1],up_limit),(0,0,255),2)
                        # Draw inferior limit
                        img2 = cv2.rectangle(img2,(0,int(img.shape[0]-down_limit)),(img.shape[1], img.shape[0]),(0,0,0), -1)
                        img = cv2.line(img,(0, int(img.shape[0]-down_limit)),(img.shape[1],int(img.shape[0]-down_limit)),(0,0,255),2)
                        # Overlay image
                        img = cv2.addWeighted(src2 = img2, src1 = img, 
                            alpha = 0.2, beta = 1, gamma = 0)
                    img = cv2.resize(img, 
                        (int(img.shape[1]*sc_fc), int(img.shape[0]*sc_fc)))
                    angle = data[idx]['steering']
                    if label == cam_labels[0]: angle += CORRECTION
                    elif label == cam_labels[2]: angle -= CORRECTION
                    angle = round(angle, 2)
                    print_list_text(
                        img, ("Angle: {}[rad]".format(angle), "cam: {}".format(label)), 
                        origin=(10, img.shape[0] - 35), color=(255, 0, 255), 
                        thickness=1, fontScale=0.50,  y_space=20)
                    imgs.append(img)

                # Concatenate all images
                img = np.concatenate(imgs, axis = 1)

                str_list = ("A: stop/reproduce", "Q: Quit", "{}%".format(round(idx*100./len(data), 2)))
                print_list_text(
                    img, str_list, origin=(10, 15), color=(0, 255, 255), 
                    thickness=1, fontScale=0.50,  y_space=20)
                cv2.imshow("data_visualization", img)

                # Increment index
                idx =-1 if loop and idx == len(data) -1 else (idx+1 if reproduce else idx)
            
            user_inpu = cv2.waitKey(int(1000/fps)) & 0xFF
            if user_inpu == ord('q') or  user_inpu == ord('Q'): break
            if user_inpu == ord('a') or  user_inpu == ord('A'): reproduce = not reproduce

        if user_inpu == ord('q') or user_inpu == ord('Q'): break
        if idx >= len(data) -1: break
    
    # Destroy GUI/User interface window
    cv2.destroyAllWindows()

def plot_data_distribution(y_data, scfc=30, graph_name="", save_name=None):

    """ plots the dataset steering angle distribution 
    Args:
        y_data: `list` of steering angles
        scfc: `int` graph's bars scaling factor
        graph_name: `string` graphs name
        save_name" `string` absolute path to save graph
    Returns: 
    """

    plt.figure(figsize=(15,5))
    plt.hist(y_data, density=False, bins= [bin/scfc for bin in range(-scfc, scfc, 1)], facecolor='g', alpha=0.75)
    plt.xlabel('Steering Angles')
    plt.ylabel('Distributions')
    plt.title(graph_name)
    plt.grid(True)
    if save_name is not None:
        plt.savefig(save_name)
    plt.show()

def plot_training_history(history_object, fig_size=(20,30), save_name=None):

    """ plots the training loss 
    Args:
        history_object: `dic` with training history loss for validation and training
        fig_size: `tuple` graph's size
        save_name" `string` absolute path to save graph
    Returns: 
    """

    plt.figure(figsize=fig_size)
    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.grid(True)
    if save_name is not None:
            plt.savefig(save_name)
    plt.show()

def create_model_result_video(src_path, dst_path="", save_name="model_results_cam_c.mp4", 
    fps=30., video_size=(360, 160)):
    
    """ creates models results from behavioral driving images 
    Args:
        src_path: `string` absolute path where models results images are stored
        dst_path" `string` destination path to save video
        save_name" `string` video file name
        fps" `float` video frame rate
        video_size" `tuple` desired video size
    Returns: 
    """

    # Check for images extensions
    extensions = ["jpg","gif","png","tga"]

    # Load images in 'path'
    imgs_path_list = [item for i in [glob.glob(src_path+'/*.%s' % ext) for ext in extensions] for item in i]
    
    # Define the codec and format and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec H264, format MP4
    video_name = os.path.join(dst_path, save_name) # File name and format
    video_out = cv2.VideoWriter(video_name, fourcc, fps, video_size) # Video recorder variable

    for imgs_path in imgs_path_list:
        
        # Read image and write it to video
        img = cv2.imread(os.path.join(imgs_path))
        video_out.write(cv2.resize(img, video_size))

    # Release video variable memory
    video_out.release()

# =============================================================================