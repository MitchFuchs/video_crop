import cv2
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import os.path
from pathlib import Path
import PIL.Image, PIL.ImageTk
import pandas as pd
import numpy as np
import time

class Cropper:
    def __init__(self):
        # self.canvas_w, self.canvas_h = 640, 360
        self.canvas_w, self.canvas_h = 1280, 720
        self.last = None
        self.vid = None
        self.out = None
        self._job = None
        self.newFileName = None
        self.selected_dir = ''
        self.selected_dir = "/home/mitch/clip_extractor/Push"
        # self.selected_dir = "/mnt/data/chimps/Directional_Push_2"
        self.files = []
        self.csv_name = "crop.txt"
        self.df = pd.DataFrame()
        self.selected_dirs = []
        self.bt_width = 15
        # self.bt_width = 20
        self.divider = 1
        self.linux_multiplier = 2
        self.delay = 50
        self.video_speed = 50
        self.prev_timestamp = 0
        self.ret = False
        self.cropping = False
        self.cropped = False
        self.clicked = False
        self.x_start, self.y_start, self.x_end, self.y_end = 0, 0, 0, 0
        self.rect_w, self.rect_h = 0, 0
        self.res_w, self.res_h = 0, 0
        self.photo = None
        self.cropped_frame = None
        self.cropped_img = None
        self.col_names = ['file', 'x_start', 'x_end', 'y_start', 'y_end', 'keep']

        self.root = Tk()
        self.root.configure(background='light grey')
        self.root.title("Video cropper")

        # create a label, entrybox and button for 'folder'
        Label(self.root, text="Choose folder", background='light grey').grid(row=1, column=1)
        self.entry_folder = Entry(self.root, width=50)
        self.entry_folder.grid(row=2, column=1, padx=30)
        self.entry_folder.insert(0, self.selected_dir)
        self.bt_folder = Button(self.root, text="Browse", width=self.bt_width,
                                     command=lambda: self.ask_directory("<Button-1>"))
        self.bt_folder.grid(row=3, column=1, pady=10)

        self.lab_asp = Label(self.root, text="", font=('', 25), background='light grey')
        self.lab_asp.grid(row=4, column=1, pady=10)
        self.lab_res = Label(self.root, text="", font=('', 25), background='light grey')
        self.lab_res.grid(row=5, column=1, pady=10)
        self.lab_pos = Label(self.root, text="", font=('', 25), background='light grey')
        self.lab_pos.grid(row=6, column=1)

        # self.label_resolution = Label(self.root, text="choose your resolution", background='light grey')
        # self.label_resolution.grid(row=7, column=1, pady=5)

        self.frame = LabelFrame(self.root, text="choose your resolution", bg='light grey', pady=5)
        self.var = StringVar()
        self.ratios = ['16:9', '4:3', '1:1']
        self.res16_9 = ['1920x1080', '1280x720', '1024x576', '960x540', '854x480', '640x360', '512x288', '256x144']
        self.res4_3 = ['1440x1080', '1280x960', '1024x768', '960x720', '800x600', '640x480', '320x240']
        self.res1_1 = ['1280x1280', '1080x1080', '960x960', '720x720', '640x640', '480x480', '360x360']
        self.resolutions = [self.res16_9, self.res4_3, self.res1_1]
        self.all_resolutions = []
        for i, resolution in enumerate(self.resolutions):
            for j, res in enumerate(resolution):
                Radiobutton(self.frame, text=res, background='light grey', variable=self.var, value=res,
                    command=lambda: self.change_rect_size()).grid(row=j, column=i+1, padx=5, pady=5, sticky=W)
                self.all_resolutions.append(res)
        self.var.set(self.all_resolutions[-1])
        self.frame.grid(row=8, column=1)
        self.frame2 = LabelFrame(self.root,  text="Clip", bg='light grey', pady=15)
        self.bt_ignore = Button(self.frame2, text="Ignore", bg='light grey', activebackground='light grey', width=self.bt_width, command=self.ignore)
        self.bt_ignore.grid(row=1, column=1)
        self.frame2.grid(row=12, column=1, columnspan=1)

        # create canvas for video
        self.canvas = Canvas(self.root, width=self.canvas_w, height=self.canvas_h)
        self.canvas.grid(row=1, column=5, rowspan=8, columnspan=4, pady=5)
        self.canvas.bind('<Motion>', self.mouse)
        self.canvas.bind("<Button-1>", self.mouse)
        self.img = self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
        self.rect = self.canvas.create_rectangle(0, 0, 0, 0, outline='red')
        self.blue_rect = self.canvas.create_rectangle(0, 0, 0, 0, outline='blue')

        self.bt_first = Button(self.root, text="First", width=self.bt_width, command=self.select_first)
        self.bt_first.grid(row=10, column=5, pady=5)
        self.bt_prev = Button(self.root, text="Previous", width=self.bt_width, command=self.select_prev)
        self.bt_prev.grid(row=10, column=6, pady=5)
        self.bt_next = Button(self.root, text="Next", width=self.bt_width, command=self.select_next)
        self.bt_next.grid(row=10, column=7, pady=5)
        self.bt_last = Button(self.root, text="Last", width=self.bt_width, command=self.select_last)
        self.bt_last.grid(row=10, column=8, pady=5)

        self.label_videos = Label(self.root, text="", background='light grey')
        self.label_videos.grid(row=11, column=5, columnspan=2, pady=5)

        self.bt_export = Button(self.root, text="Export crops", width=self.bt_width, command=self.export)
        self.bt_export.grid(row=11, column=7, pady=5)

        self.bt_save = Button(self.root, text="Save", width=self.bt_width, command=self.save)
        self.bt_save.grid(row=11, column=8, pady=5)

        self.preview = Canvas(self.root, width=self.canvas_w, height=self.canvas_h)
        self.preview.grid(row=12, column=5, rowspan=9, columnspan=5, pady=5)
        self.thumbnail = self.preview.create_image(0, 0, image=self.cropped_frame, anchor=NW)

        self.bindings()
        self.change_rect_size()
        self.move_blue_rect()
        self.update_label_res()
        if len(self.selected_dir) > 0:
            self.boot()
        self.root.mainloop()

    def boot(self):
        self.load_files()
        self.load_csv()
        self.select_first()

    def bindings(self):
        self.root.bind('r', lambda event: self.change_resolution('next'))
        self.root.bind('e', lambda event: self.change_resolution('previous'))
        self.root.bind('n', lambda event: self.select_next())
        self.root.bind('b', lambda event: self.select_prev())
        self.root.bind('l', lambda event: self.select_last())
        self.root.bind('f', lambda event: self.select_first())

    def change_resolution(self, direction):
        if direction == 'next':
            ind = self.all_resolutions.index(self.var.get())+1
            if ind == len(self.all_resolutions): ind = 0
        elif direction == 'previous':
            ind = self.all_resolutions.index(self.var.get())-1
            if ind < 0:
                ind = len(self.all_resolutions)-1
        self.var.set(self.all_resolutions[ind])
        self.change_rect_size()
        self.move_blue_rect()

    def change_rect_size(self):
        self.rect_w, self.rect_h = [int(x) for x in self.var.get().split('x')]
        self.res_w, self.res_h = self.rect_w / self.divider, self.rect_h / self.divider
        self.update_label_res()

    def mouse(self, event):
        x, y = event.x, event.y

        x_start, x_end = x - self.res_w / 2, x + self.res_w / 2
        y_start, y_end = y - self.res_h / 2, y + self.res_h / 2

        if x_start < 0:
            x_start, x_end = 0, self.res_w
        elif x_end > self.canvas_w:
            x_start, x_end = self.canvas_w - self.res_w, self.canvas_w
        if y_start < 0:
            y_start, y_end = 0, self.res_h
        elif y_end > self.canvas_h:
            y_start, y_end = self.canvas_h - self.res_h, self.canvas_h

        if event.type == EventType.Motion:
            self.canvas.coords(self.blue_rect, x_start, y_start, x_end, y_end)
            if self.cropped:
                self.move_rect()
        elif event.type == EventType.ButtonPress:
            # self.canvas.coords(self.rect, x_start, y_start, x_end, y_end)
            # convert to image coordinates
            self.x_start, self.x_end = x_start * self.divider, x_end * self.divider
            self.y_start, self.y_end = y_start * self.divider, y_end * self.divider
            self.df.loc[self.vid.video_source] = [self.x_start, self.x_end, self.y_start, self.y_end]
            self.cropped = True
            self.clicked = True
            self.move_rect()
            self.cancel()
            self.play_preview()

    def play_preview(self):
        time_elapsed = time.time() - self.prev_timestamp
        if time_elapsed > 1. / self.video_speed:
            if self.vid.vid.get(cv2.CAP_PROP_POS_FRAMES) == 0:
                self.vid = MyVideoCapture(0, self.selected_dir, self.files, self.vid.index)
                self.x_start, self.x_end, self.y_start, self.y_end = self.df.loc[self.vid.video_source][:-1]
            ret, frame = self.vid.get_cropped_frame(self.x_start, self.x_end, self.y_start, self.y_end)
            self.prev_timestamp = time.time()
            if ret:
                aspect_ratio = frame.shape[0] / frame.shape[1]
                new_size = (int(self.canvas_h / aspect_ratio), self.canvas_h)
                frame = cv2.resize(frame, dsize=new_size)
                self.cropped_frame = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
                self.preview.itemconfigure(self.thumbnail, image=self.cropped_frame)
            else:
                self.vid.vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._job = self.root.after(self.delay, self.play_preview)

    def move_rect(self):
        self.canvas.coords(self.rect,
                           self.df.loc[self.vid.video_source, 'x_start'] / self.divider,
                           self.df.loc[self.vid.video_source, 'y_start'] / self.divider,
                           self.df.loc[self.vid.video_source, 'x_end'] / self.divider,
                           self.df.loc[self.vid.video_source, 'y_end'] / self.divider)

    def move_blue_rect(self):
        rect_coords = self.canvas.coords(self.blue_rect)
        rect_center_x, rect_center_y = (rect_coords[0] + rect_coords[2]) / 2, (rect_coords[1] + rect_coords[3]) / 2
        self.canvas.coords(self.blue_rect,
                           rect_center_x - self.res_w / 2,
                           rect_center_y - self.res_h / 2,
                           rect_center_x + self.res_w / 2,
                           rect_center_y + self.res_h / 2)

    def reset_rect(self):
        self.canvas.coords(self.rect, 0, 0, 0, 0)

    def reset_blue_rect(self):
        self.canvas.coords(self.blue_rect, 0, 0, 0, 0)

    def reset_ignore(self):
        # only if TRUE specified in df
        if not pd.isnull(self.df.loc[self.vid.video_source, 'keep']) and self.df.loc[self.vid.video_source, 'keep']:
            self.ignore_red()
        else:
            self.ignore_grey()

    def reset_preview(self):
        self.preview.delete(self.thumbnail)
        self.thumbnail = self.preview.create_image(0, 0, image=None, anchor=NW)
        self.preview.itemconfigure(self.thumbnail, image=None)

    def update_canvas(self):
        if self.vid.ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.vid.first_frame))
            self.canvas.itemconfigure(self.img, image=self.photo)
        self.update_label_videos()

    def export(self):
        file = os.path.join(self.selected_dir, self.csv_name)
        df = pd.read_csv(file, header=0, names=self.col_names, index_col=0)
        df.dropna(axis=0, how='all', subset=['x_start'], inplace=True)
        for index, row in df.iterrows():
            vid = MyVideoCapture(0, self.selected_dir, self.files, self.files.index(index))
            filename = os.path.join(str(Path(vid.videofile).parents[0]), str(Path(vid.videofile).stem) +'_cropped.avi')
            width = int(row['x_end'] - row['x_start'])
            height = int(row['y_end'] - row['y_start'])
            out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), vid.fps, (width, height))
            ret, _ = vid.vid.read()
            while ret:
                ret, frame = vid.vid.read()
                if ret:
                    cropped = frame[int(row['y_start']):int(row['y_end']), int(row['x_start']):int(row['x_end'])]
                    out.write(cropped)
            print("New cropped video created: ", filename)
            vid.__del__()
            out.release()
        cv2.destroyAllWindows()

    def write_video(self, vid, ):
        self.newFileName = os.path.join(str(Path(vid.videofile).parents[0]), str(Path(vid.videofile).stem) +'_cropped.avi')
        self.out = cv2.VideoWriter(self.newFileName, cv2.VideoWriter_fourcc('M','J','P','G'), self.vid.fps, (self.rect_w, self.rect_h))
        self.ret, _ = self.vid.vid.read()
        while self.ret:
            self.ret, self.frame = self.vid.vid.read()
            if self.ret:
                self.cropped_img = self.frame[self.y_start:self.y_end, self.x_start:self.x_end]
                self.out.write(self.cropped_img)
        self.vid.__del__()
        self.out.release()
        cv2.destroyAllWindows()
        print("New cropped video created: ", self.newFileName)

    def ask_directory(self, event):
        self.selected_dir = filedialog.askdirectory()
        self.entry_folder.insert(0, self.selected_dir)
        # if self.selected_dir not in self.selected_dirs:
        #     self.selected_dirs.insert(0, self.selected_dir)
        self.boot()

    def load_csv(self):
        file = os.path.join(self.selected_dir, self.csv_name)
        if not os.path.isfile(file):
            with open(file, "w") as f:
                for vid_name in self.files:
                    f.write(vid_name + "\n")
            print('crop file created')
            self.df = pd.read_csv(file, header=None, names=self.col_names, index_col=0)
        else:
            df = pd.read_csv(file, header=0, names=self.col_names, index_col=0)
            files_to_add = [x for x in self.files if x not in df.index]
            df_to_add = pd.DataFrame(data=[], columns=self.col_names[1:], index=files_to_add)
            df_to_add = df_to_add.rename_axis('file')
            self.df = pd.concat([df, df_to_add])

    def save(self):
        self.df.to_csv(os.path.join(self.selected_dir, self.csv_name))
        self.load_csv()

    def load_files(self):
        self.files = [f for f in os.listdir(self.selected_dir) if f.endswith('.mp4')]

    def cancel(self):
        if self._job is not None:
            self.root.after_cancel(self._job)
            self._job = None

    def reset_after_video_change(self):
        self.cancel()
        self.reset_preview()
        self.reset_rect()
        self.reset_blue_rect()
        self.reset_ignore()
        self.clicked = False
        self.divider = self.vid.width / self.canvas_w
        x_start = self.df.loc[self.vid.video_source, 'x_start']
        if np.isnan(x_start):
            self.cropped = False
        elif x_start >= 0:
            self.cropped = True
            self.crop_size_from_df()
            self.move_rect()
            self.play_preview()
        self.update_canvas()

    def crop_size_from_df(self):
        x_start, y_start, x_end, y_end = \
            self.df.loc[self.vid.video_source, ['x_start', 'y_start', 'x_end', 'y_end']]
        w = round(x_end - x_start)
        h = round(y_end - y_start)
        self.var.set(str(w)+'x'+str(h))
        self.change_rect_size()

    def select_first(self):
        self.vid = MyVideoCapture(0, self.selected_dir, self.files, 0)
        self.reset_after_video_change()

    def select_last(self):
        self.vid = MyVideoCapture(0, self.selected_dir, self.files, len(self.files)-1)
        self.reset_after_video_change()

    def select_next(self):
        self.vid = MyVideoCapture(0, self.selected_dir, self.files, min(len(self.files)-1, self.vid.index+1))
        self.reset_after_video_change()

    def select_prev(self):
        self.vid = MyVideoCapture(0, self.selected_dir, self.files, max(0, self.vid.index-1))
        self.reset_after_video_change()

    def ignore(self):
        if self.vid is not None:
            if self.bt_ignore['text'] == 'Ignore':
                self.ignore_red()
                self.df.loc[self.vid.video_source, 'keep'] = True
            else:
                self.ignore_grey()
                self.df.loc[self.vid.video_source, 'keep'] = False

    def ignore_red(self):
        self.bt_ignore.config(text='Ignored', background='red', activebackground='red')

    def ignore_grey(self):
        self.bt_ignore.config(text='Ignore', background='light grey', activebackground='light grey')

    def update_label_videos(self):
        self.label_videos.config(text=f'{self.vid.index+1}/{len(self.files)} : video_name: {self.vid.video_source}')

    def update_label_res(self):
        res = self.var.get()
        ind = self.resolutions.index([x for x in self.resolutions if res in x][0])
        self.lab_res.config(text=f'{res}')
        self.lab_asp.config(text=f'{self.ratios[ind]}')

class MyVideoCapture:
    def __init__(self, start_milli, directory, files, index):
        # Open the video source
        self.linux_multiplier = 2
        self.dim = (640*self.linux_multiplier, 360*self.linux_multiplier)
        self.start_milli = start_milli
        self.directory = directory
        self.files = files
        self.index = index
        self.video_source = self.files[self.index]
        self.videofile = os.path.join(self.directory, self.video_source)
        self.vid = cv2.VideoCapture(self.videofile)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", self.video_source)
        self.ret, self.first_frame = self.vid.read()
        self.first_frame = cv2.resize(cv2.cvtColor(self.first_frame, cv2.COLOR_BGR2RGB), self.dim)
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # self.dim = self.get_dim(self.width, self.height)
        self.fps = 25

    def get_frame(self):
        while self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return ret, cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), self.dim)
            else:
                return ret, None
        else:
            return False, None

    def get_cropped_frame(self, x_start, x_end, y_start, y_end):
        width = int(x_end - x_start)
        height = int(y_end - y_start)
        # print(width, height)
        while self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                cropped = frame[int(y_start):int(y_end), int(x_start):int(x_end)]
                # print(cropped.shape)
                # Return a boolean success flag and the current frame converted to BGR
                return ret, cv2.resize(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB), (width, height))
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
