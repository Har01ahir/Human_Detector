import cv2
import imutils
import numpy as np
import argparse

def detection(frame):
    b_box=H_Cascade.detectMultiScale(frame,1.25,3,2,[25,25])
    human = 1
    for x,y,w,h in b_box:
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if w>2 and h>2:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, f'human {human}', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            human += 1 
    
    cv2.putText(frame, 'Detecting..... ', (40,40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
    cv2.imshow('output', frame)

    return frame


def detecting_Vdo(path, writer):

    vdo = cv2.VideoCapture(path)
    checking, frame = vdo.read()
    if checking == False:
        print('Video Not Found. Please Enter a Valid Path (Full path of Video Should be Provided).')
        return

    print('Detecting people...')
    while vdo.isOpened():
        checking, frame =  vdo.read()

        if checking:
            frame = imutils.resize(frame ,width=min(800, frame.shape[1]))
            frame = detection(frame)
            
            if writer is not None:
                writer.write(frame)
            
            key = cv2.waitKey(1)
            if key== ord('q'):
                break
        else:
            break
    vdo.release()
    cv2.destroyAllWindows()

def detecting_WebCam(writer):   
    vdo = cv2.VideoCapture(0)
    print('Detecting people...')

    while True:
        checking, frame = vdo.read()

        frame = detection(frame)
        if writer is not None:
            writer.write(frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
                break

    vdo.release()
    cv2.destroyAllWindows()


def detecting_Img(path, out_):
    # img = cv2.imread(path)
    # img = imutils.resize(img, width = min(800, img.shape[1])) 

    # detected_img = detection(img)

    # if out_ is not None:
    #     cv2.imwrite(out_, detected_img)
        
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    for i in range(0,559):
        # print(path)
        img = cv2.imread(path+f'{i}.png')
        img = imutils.resize(img, width = min(800, img.shape[1])) 

        detected_img = detection(img)

        if out_ is not None:
            cv2.imwrite(out_, detected_img)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if i%10==0 and i!=0:
            print('Enter 1 to exit')
            a=input('__')
            try:
                if int(a)==1:
                    break
            except:
                print('invalid input')


def Detect_human(args):
    Img=args["image"]
    Vdo=args['video']
    out_=args['output']
    writer = None
    if args['output'] is not None and Img is None:
        writer = cv2.VideoWriter(args['output'],cv2.VideoWriter_fourcc(*'MJPG'), 10, (600,600))

    if str(args["camera"]) == 'true':
        print('[INFO] Opening Web Cam.')
        detecting_WebCam(writer)
    elif Vdo is not None:
        path=args['video']
        print('[INFO] Opening Video from path.')
        detecting_Vdo(path, writer)
    elif Img is not None:
        path = Img
        print('[INFO] Opening Image from path.')
        detecting_Img(path, out_)

def argsParser():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-v", "--video", default=None, help="path to Video File ")
    arg_parse.add_argument("-i", "--image", default=None, help="path to Image File ")
    arg_parse.add_argument("-c", "--camera", default=None, help="Set true if you want to use the camera.")
    arg_parse.add_argument("-o", "--output", type=str, help="path to optional output video file")
    args = vars(arg_parse.parse_args())

    return args

if __name__ == "__main__":
    H_Cascade=cv2.CascadeClassifier('D:\Hardik\MyImp\MyDocs\project\haarcascade_fullbody.xml')
    args = argsParser()
    Detect_human(args)

