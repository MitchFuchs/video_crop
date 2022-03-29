# video_cropping
his GUI provides a tool to crop video in a specified directory to different resolutions. 
Choose from 22 resolutions in 3 different aspect ratios: 16:9. 4:3 or 1:1. Once all cropped region are defined, export the new clips with 'Export crops'.

# demo 
![video_cropper](https://user-images.githubusercontent.com/73831423/160564419-1c1cf267-a315-43ff-b839-22426f439a13.JPG)

# use case example 
You need to crop videos in predefined resolutions to recreate smaller videos from a region

# instructions
1) After starting the application, choose the directory in which all your videos are stored. 
2) You can then go through your videos, using the 'Next', 'Previous', 'First' and 'Last' buttons. 
3) For each video, select the resolution, you want to extract. Resolutions in the first coloumn have an aspect ratio of 16:9, second column 4:3 and third column 1:1
5) Move your mouse over the video and place the red rectangle, on the region you want to extract. 
6) When your happy with the region, clic it with your mouse. The preview panel shows you the extracted region. 
7) Repeat for all videos and once you are done, clic 'export crops' to extract in new files. 
8) You can always pause your selection process by clicking 'Save' and coming back to it later. 

<!-- # dependencies

Using our env freeze: 
- `pip install -r requirements.txt`

Or manually install:
- `[pip or conda] install opencv`
- `[pip or conda] install pillow`
- `[pip or conda] install pandas`
- `[pip or conda] install xlsxwriter`

# dev

To freeze the venv:
- `pip list --format=freeze > requirements.txt`

To build executable on mac (app witll be in build folder):
- `pyinstaller --onefile --hidden-import cmath main.py`
 -->
# contact
Feel free to reach out at michael@neptune-consulting.ch
Cheers, 
Mitch. 
