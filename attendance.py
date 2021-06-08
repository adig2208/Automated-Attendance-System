from tkinter import*
import tkinter as tk
import cv2,os
import csv
import numpy as np
from PIL import Image
from PIL import ImageTk
import pandas as pd
import datetime
import time
from tkinter import messagebox


window = tk.Tk()

screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
#set width and height

canvas=Canvas(window,width=screen_width,height=screen_height)
canvas.pack(expand=YES, fill=BOTH)

#give this image path. image should be in png format.

image=PhotoImage(file="F:\ADVAIT\ADVAIT RESUME\IIC\Bg_image.png")

canvas.create_image(0,0,anchor=NW,image=image)
canvas.pack()

window.title("Attendance System")

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

x_cord = 150;
y_cord = 100;
checker=0;
message = tk.Label(window, text="*corona precautionary norms followed" ,bg="darkblue"  ,fg="white"  ,width=40  ,height=1,font=('Times New Roman', 13, 'normal')) 
message.place(x=1000, y=5)

message = tk.Label(window, text="AUTOMATED ATTENDACE SYSTEM" ,bg ="darkblue"  ,fg="white"  ,width=30  ,height=1,font=('Times New Roman', 25, 'bold underline')) 
message.place(x=350, y=5)

lbl = tk.Label(window, text="Enter Your College ID  :",width=20  ,height=1  ,fg="white"  ,bg="darkblue" ,font=('Times New Roman', 25, ' bold ') ) 
lbl.place(x=230-x_cord, y=228-y_cord)

txt = tk.Entry(window,width=30,bg="white" ,fg="black",font=('Times New Roman', 15, ' bold '))
txt.place(x=650-x_cord, y=237-y_cord)

lbl2 = tk.Label(window, text="Enter Your Name   :",width=20  ,fg="white"  ,bg="darkblue"    ,height=1 ,font=('Times New Roman', 25, ' bold ')) 
lbl2.place(x=230-x_cord, y=378-y_cord)

txt2 = tk.Entry(window,width=30  ,bg="white"  ,fg="black",font=('Times New Roman', 15, ' bold ')  )
txt2.place(x=650-x_cord, y=387-y_cord)

lbl3 = tk.Label(window, text="NOTIFICATION    :",width=20  ,fg="white"  ,bg="darkblue"  ,height=1 ,font=('Times New Roman', 25, ' bold ')) 
lbl3.place(x=230-x_cord, y=548-y_cord)

message = tk.Label(window, text="" ,bg="white"  ,fg="blue"  ,width=30  ,height=1, activebackground = "white" ,font=('Times New Roman', 15, ' bold ')) 
message.place(x=650-x_cord, y=557-y_cord)

lbl3 = tk.Label(window, text="ATTENDANCE",width=20  ,fg="white"  ,bg="black"  ,height=2 ,font=('Times New Roman', 30, ' bold ')) 
lbl3.place(x=120, y=700-y_cord)


message2 = tk.Label(window, text="" ,fg="white"   ,bg="black",activeforeground = "green",width=40  ,height=4 ,font=('times', 15, ' bold ')) 
message2.place(x=700, y=700-y_cord)

def clear1():
    txt.delete(0, 'end')    
    res = ""
    message.configure(text= res)

def clear2():
    txt2.delete(0, 'end')    
    res = ""
    message.configure(text= res)    
    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False
 

def mask_detection():
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.models import load_model
    from imutils.video import VideoStream
    import numpy as np
    import argparse
    import imutils
    import time
    import cv2
    import os

    def detect_and_predict_mask(frame, faceNet, maskNet):
        # grab the dimensions of the frame and then construct a blob
        # from it
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (200, 200),
            (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        faceNet.setInput(blob)
        detections = faceNet.forward()

        # initialize our list of faces, their corresponding locations,
        # and the list of predictions from our face mask network
        faces = []
        locs = []
        preds = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > args["confidence"]:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

        # only make a predictions if at least one face was detected
        if len(faces) > 0:
            # for faster inference we'll make batch predictions on *all*
            # faces at the same time rather than one-by-one predictions
            # in the above `for` loop
            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=32)

        # return a 2-tuple of the face locations and their corresponding
        # locations
        return (locs, preds)

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--face", type=str,
        default="face_detector",
        help="path to face detector model directory")
    ap.add_argument("-m", "--model", type=str,
        default="mask_detector.model",
        help="path to trained face mask detector model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
    weightsPath = os.path.sep.join([args["face"],
        "res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    maskNet = load_model(args["model"])

    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=600)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()



def TakeImages():        
    Id=(txt.get())
    name=(txt2.get())
    if not Id:
        res="Please enter Id"
        message.configure(text = res)
        MsgBox = tk.messagebox.askquestion ("Warning","Please enter roll number properly , press yes if you understood",icon = 'warning')
        if MsgBox == 'no':
            tk.messagebox.showinfo('Your need','Please go through the readme file properly')
    elif not name:
        res="Please enter Name"
        message.configure(text = res)
        MsgBox = tk.messagebox.askquestion ("Warning","Please enter your name properly , press yes if you understood",icon = 'warning')
        if MsgBox == 'no':
            tk.messagebox.showinfo('Your need','Please go through the readme file properly')
        
    elif(is_number(Id) and name.isalpha()):
            cam = cv2.VideoCapture(0)
            harcascadePath = "haarcascade_frontalface_default.xml"
            detector=cv2.CascadeClassifier(harcascadePath)
            sampleNum=0
            while(True):
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                for (x,y,w,h) in faces:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
                    #incrementing sample number 
                    sampleNum=sampleNum+1
                    #saving the captured face in the dataset folder TrainingImage
                    cv2.imwrite("TrainingImage\ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                    #display the frame
                    cv2.imshow('frame',img)
                #wait for 100 miliseconds 
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
                # break if the sample number is morethan 100
                elif sampleNum>60:
                    break
            cam.release()
            cv2.destroyAllWindows() 
            res = "Images Saved for ID : " + Id +" Name : "+ name
            row = [Id , name]
            with open('StudentDetails\StudentDetails.csv','a+') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
            csvFile.close()
            message.configure(text= res)
    else:
        if(is_number(Id)):
            res = "Enter Alphabetical Name"
            message.configure(text= res)
        if(name.isalpha()):
            res = "Enter Numeric Id"
            message.configure(text= res)
            
    
def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    faces,Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res = "Image Trained"
    clear1();
    clear2();
    message.configure(text= res)
    tk.messagebox.showinfo('Completed','Your model has been trained successfully!!')
    

def getImagesAndLabels(path):

    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    
    faces=[]

    Ids=[]

    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)        
    return faces,Ids

def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()#cv2.createLBPHFaceRecognizer()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
    df=pd.read_csv("StudentDetails\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    col_names =  ['Id','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)    
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])                                   
            if(conf < 50):
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]
                
            else:
                Id='Unknown'                
                tt=str(Id)  
            if(conf > 75):
                noOfFile=len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')    
        cv2.imshow('im',im) 
        if (cv2.waitKey(1)==ord('q')):
            break
    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName,index=False)
    cam.release()
    cv2.destroyAllWindows()
    res=attendance
    message2.configure(text= res)
    res = "Attendance Taken"
    message.configure(text= res)
    tk.messagebox.showinfo('Completed','Congratulations ! Your attendance has been marked successfully for the day!!')
    
def quit_window():
   MsgBox = tk.messagebox.askquestion ('Exit Application','Are you sure you want to exit the application',icon = 'warning')
   if MsgBox == 'yes':
       tk.messagebox.showinfo("Greetings", "Thank You very much for using our software. Have a nice day ahead!!")
       window.destroy()

    
# takeImg = tk.Button(window, text="MASK DETECTION", command=mask_detection  ,fg="white"  ,bg="blue"  ,width=25  ,height=2, activebackground = "pink" ,font=('Times New Roman', 15, ' bold '))
# takeImg.place(x=0, y=0)


lbl4 = tk.Label(window, text="REGISTER",width=15 ,fg="white"  ,bg="darkblue"  ,height=1 ,font=('Times New Roman', 20, ' bold '))
lbl4.place(x=1145-x_cord, y=200-y_cord)
  
takeImg = tk.Button(window, text="CAPTURE IMAGE", command=TakeImages  ,fg="white"  ,bg="black"  ,width=25  ,height=2, activebackground = "white" ,font=('Times New Roman', 15, ' bold '))
takeImg.place(x=1110-x_cord, y=250-y_cord)
trainImg = tk.Button(window, text="TRAIN  MODEL", command=TrainImages  ,fg="white"  ,bg="black"  ,width=25  ,height=2, activebackground = "white" ,font=('Times New Roman', 15, ' bold '))
trainImg.place(x=1110-x_cord, y=320-y_cord)

lbl5 = tk.Label(window, text="ATTENDANCE",width=15  ,fg="white"  ,bg="darkblue"  ,height=1 ,font=('Times New Roman', 20, ' bold ')) 
lbl5.place(x=1145-x_cord, y=430-y_cord)


trackImg = tk.Button(window, text="TEMPERATURE", command="" ,fg="white"  ,bg="black"  ,width=25  ,height=2, activebackground = "white" ,font=('Times New Roman', 15, ' bold '))
trackImg.place(x=1110-x_cord, y=480-y_cord)
trackImg = tk.Button(window, text="DETECT MASK", command=mask_detection ,fg="white"  ,bg="black"  ,width=25  ,height=2, activebackground = "white" ,font=('Times New Roman', 15, ' bold '))
trackImg.place(x=1110-x_cord, y=550-y_cord)
trackImg = tk.Button(window, text="MARK ATTENDANCE", command=TrackImages  ,fg="white"  ,bg="black"  ,width=25  ,height=2, activebackground = "white" ,font=('Times New Roman', 15, ' bold '))
trackImg.place(x=1110-x_cord, y=620-y_cord)
quitWindow = tk.Button(window, text="QUIT", command=quit_window  ,fg="white"  ,bg="red"  ,width=10 ,height=2, activebackground = "pink" ,font=('Times New Roman',14, ' bold '))
quitWindow.place(x=1220, y=740-y_cord)
 
window.mainloop()