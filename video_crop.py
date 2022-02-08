"""
Author: Stefan Palm
stefan.palm@hotmail.se

code is intended for learning
"""

import cv2
import tkinter as tk
from tkinter import filedialog
import os.path
from pathlib import Path


class Cropper:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()

        #catching some info from root for later
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()

        # Select the file to work on, need to use withdraw to get rid of dialogue window
        self.videofile = filedialog.askopenfilename()

        #Let us open the Video and start reading

        self.vid = MyVideoCapture(0, self.videofile)
        self.ret, self.frame = self.vid.get_frame()

        #prep some stuff for cropping
        self.cropping = False
        self.x_start, self.y_start, self.x_end, self.y_end = 0, 0, 0, 0
        self.oriImage = self.frame.copy()

        #prep a window, and place it at the center of the screen
        cv2.namedWindow("select area, use q when happy")
        cv2.setMouseCallback("select area, use q when happy", self.mouse_crop)
        cv2.resizeWindow('select area, use q when happy', self.frame.shape[1], self.frame.shape[0])
        self.x_pos = round(self.screen_width/2) - round(self.frame.shape[1]/2)
        self.y_pos = round(self.screen_height/2) - round(self.frame.shape[0]/2)
        cv2.moveWindow("select area, use q when happy", self.x_pos, self.y_pos)

        #show image and let user crop, press q when happy
        while self.ret:
            i = self.frame.copy()
            if not self.cropping:
                cv2.imshow("select area, use q when happy", self.frame)
                if (self.x_start + self.y_start + self.x_end + self.y_end) > 0:
                    cv2.rectangle(i, (self.x_start, self.y_start), (self.x_end, self.y_end), (255, 0, 0), 2)
                    cv2.imshow("select area, use q when happy", i)

            elif self.cropping:
                cv2.rectangle(i, (self.x_start, self.y_start), (self.x_end, self.y_end), (255, 0, 0), 2)
                cv2.imshow("select area, use q when happy", i)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

        #cropping by slicing original image[y:y+h, x:x+w]
        self.cropped = self.oriImage[self.y_start:self.y_end, self.x_start:self.x_end]

        # #show the result - un-comment this out if you like
        # cv2.namedWindow('cropped area', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('cropped area', self.cropped.shape[1], self.cropped.shape[0])
        # self.x_pos = round(self.screen_width/2) - round(self.cropped.shape[1]/2)
        # self.y_pos = round(self.screen_height/2) - round(self.cropped.shape[0]/2)
        # cv2.moveWindow("cropped area", self.x_pos, self.y_pos)
        # cv2.imshow("cropped area", self.cropped)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Now let us crop the video with same cropping parameters
        # self.cap = cv2.VideoCapture(self.videofile)
        self.vid = MyVideoCapture(0, self.videofile)

        #let's keep it simple...store the new file in same directory, same name, but add suffix
        self.newFileName = os.path.join(str(Path(self.videofile).parents[0]), str(Path(self.videofile).stem) +'_cropped.avi')
        self.frame_width = self.x_end - self.x_start
        self.frame_height = self.y_end - self.y_start

        # create VideoWriter object and define the codec. This may be an area for warnings and errors - use google if so
        self.out = cv2.VideoWriter(self.newFileName, cv2.VideoWriter_fourcc('M','J','P','G'), self.vid.fps, (self.frame_width, self.frame_height))

        #read frame by frame
        while self.ret:
            self.ret, self.frame = self.vid.get_frame()
            if self.ret:
                #crop frame
                self.cropped = self.frame[self.y_start:self.y_end, self.x_start:self.x_end]
                # Write the frame into the file 'output.avi'
                self.out.write(self.cropped)

                # Display the resulting frame - trying to move window, but does not always work
                cv2.namedWindow('producing video', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('producing video', self.cropped.shape[1], self.cropped.shape[0])
                self.x_pos = round(self.screen_width/2) - round(self.cropped.shape[1]/2)
                self.y_pos = round(self.screen_height/2) - round(self.cropped.shape[0]/2)
                cv2.moveWindow("producing video", self.x_pos, self.y_pos)
                cv2.imshow('producing video', self.cropped)

                # Press Q on keyboard to stop recording early
                if cv2.waitKey(1) & 0xFF == ord('q'):
                  break

        # When everything done, release the video capture and video write objects
        self.vid.__del__()
        self.out.release()

        # Make sure all windows are closed
        cv2.destroyAllWindows()

        # Leave a message
        print("New cropped video created: ", self.newFileName)

    def mouse_crop(self, event, x, y, flags, param):
        # if the left mouse button was DOWN, start RECORDING
        # (x, y) coordinates and indicate that cropping is being
        if event == cv2.EVENT_LBUTTONDOWN:
            self.x_start, self.y_start, self.x_end, self.y_end = x, y, x, y
            self.cropping = True

        # Mouse is Moving
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.cropping:
                self.x_end, self.y_end = x, y

        # if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates
            self.x_end, self.y_end = x, y
            self.cropping = False # cropping is finished

class MyVideoCapture:
    def __init__(self, start_milli, video_source):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        self.vid.set(0, int(start_milli))

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # self.dim = self.get_dim(self.width, self.height)

        # Find OpenCV version
        self.major_ver, self.minor_ver, self.subminor_ver = cv2.__version__.split('.')
        self.fps = self.vid.get(cv2.CAP_PROP_FPS)

    def get_frame(self):
        while self.vid.isOpened():

            ret, frame = self.vid.read()

            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return ret, frame
                # return ret, cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (self.width, self.height))
            else:
                return ret, None
        else:
            return False, None

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


if __name__ == "__main__":
    app = Cropper()
