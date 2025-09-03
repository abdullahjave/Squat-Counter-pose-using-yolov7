import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, scale_coords_kpt, increment_path
from utils.plots import colors, plot_one_box_kpt
from utils.torch_utils import select_device

# Calculate Angle
def calculate_angle(coords):
    a = coords[0] # Hip
    b = coords[1] # Knee
    c = coords[2] # Heel   
    
    radians = np.arctan2(c[1].item() - b[1].item(), c[0].item() - b[0].item()) - np.arctan2(a[1].item() - b[1].item(), a[0].item()-b[0].item())
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def get_hip_knee(kpts, list_id):
    steps = 3

    coord = []

    for kid in list_id:
        conf = kpts[steps * kid + 2]
        if conf < 0.5:
            break
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]

        coord.append((x_coord, y_coord))
    
    return coord

def get_coords(kpts):
    left_leg = [12, 14, 16]
    right_leg = [11, 13, 15]
    squat_pose = True
    
    # Left Foot
    coords = get_hip_knee(kpts, left_leg)

    # Right Foot
    if(len(coords) < 3):
        coords = []
        coords = get_hip_knee(kpts, right_leg)
    
    if(len(coords) < 3):
        squat_pose = False

    return coords, squat_pose

def count_squat(angle, stage, counter):       
    if angle > 160:
        if(stage == 'Down'):
            counter += 1

        stage = "Stand"            

    if angle < opt.angle and stage =='Stand' and angle != 0:
        stage="Down"

    return angle, stage, counter

def detect(opt):
    source, weights, view_img, imgsz, kpt_label = opt.source, opt.weights, opt.view_img, opt.img_size, opt.kpt_label
    save_img = not opt.nosave 

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=False))  # increment run    
    (save_dir).mkdir(parents=True, exist_ok=True)  # make dir       

    # Initialize    
    device = select_device(opt.device)    

    half = False
    if(device.type != 'cpu'):
        compute_capability = torch.cuda.get_device_capability(device=device)    
        half = (device.type != 'cpu') and (compute_capability[0] >= 8)  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model        

    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None    

    # Check Source
    webcam = source.isnumeric()
    
    if webcam:
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)        

    # Get class names
    names = model.module.names if hasattr(model, 'module') else model.names  

    # Run inference    
    t0 = time.time()

    # Squat counter variables
    counter = 0 
    stage = ""

    while(cap.isOpened()):                                 
        (grabbed, frame) = cap.read()

        if not grabbed:
            exit()

        image = frame.copy() 

        # Padded resize  
        image_pad = letterbox(image, imgsz, stride=64, auto=True)[0]
        
        # Convert
        image_trans = image_pad[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, (height x width x channels) to (channels x height x width)          
        image_trans = np.ascontiguousarray(image_trans)

        image_trans = torch.from_numpy(image_trans).to(device)
        image_trans = image_trans.half() if half else image_trans.float()  # uint8 to fp16/32        

        # Normalize
        image_trans /= 255.0  # 0 - 255 to 0.0 - 1.0

        if image_trans.ndimension() == 3:          # (batch x channels x height x width)
            image_trans = image_trans.unsqueeze(0)   

        # Inference
        t1 = time.time()
        pred = model(image_trans)[0]        
        # Apply NMS
        pred = non_max_suppression_kpt(pred, opt.conf_thres, opt.iou_thres, classes=None, agnostic=False, kpt_label=kpt_label)
        t2 = time.time()            
        
        # Detections Process
        det = pred[0]

        p = Path(source)  # to Path
        save_path = str(save_dir / p.name)  # img.jpg                        
        
        if len(det):
            # Rescale boxes from image_trans to image size
            scale_coords_kpt(image_trans.shape[2:], det[:, :4], image.shape, kpt_label=False)
            scale_coords_kpt(image_trans.shape[2:], det[:, 6:], image.shape, kpt_label=kpt_label, step=3)                                                              
            
            # Write results
            for det_index, (*xyxy, conf, cls) in enumerate(det[:,:6]): 
                                    
                if save_img or view_img:  # Add bbox to image
                    c = int(cls)  # integer class                        
                    label = None if opt.hide_labels else (f'{names[c]} {conf:.2f}')    

                    kpts = det[det_index, 6:]
                    
                    # Get Hip, Knee, Heel Coordinate
                    squat_coords = []
                    squat_coords, squat_pose = get_coords(kpts)                    

                    angle = 0
                    # Calculate Angle
                    if squat_pose:
                        angle = calculate_angle(squat_coords)                        

                    # Count Squat
                    angle, stage, counter = count_squat(angle, stage, counter)                                        

                    plot_one_box_kpt(xyxy, image, label=label, color=colors(c, True), line_thickness=opt.line_thickness, kpt_label=kpt_label, kpts=kpts, steps=3, orig_shape=image.shape[:2])                                        

            # Print time (inference + NMS)
            print(f'Done. ({t2 - t1:.3f}s)')

            # Draw Counter in Video
            # Counter 
            cv2.rectangle(image, (0,0), (300, 120), (0, 0, 0), -1)
            cv2.putText(image, "Counter : " + str(counter), (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)            
            # Stage
            stage_color = (0,255,0)
            if(stage == "Down"):
                stage_color = (0,0,255)

            cv2.putText(image, stage, (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, stage_color, 2, cv2.LINE_AA)            

            # Stream results
            if view_img:
                # Resize 
                resized = image.copy()
                if(image.shape[1] > 1500 or image.shape[0] > 1500):
                    scale_percent = 70 # percent of original size
                    width = int(image.shape[1] * scale_percent / 100)
                    height = int(image.shape[0] * scale_percent / 100)

                    dim = (width, height)
                    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

                cv2.imshow(str(p), resized)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:                
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if not webcam:  # video
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, image.shape[1], image.shape[0]
                        save_path += '.mp4'
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(image)

        if cv2.waitKey(1) == ord('q'):
            break    

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file, 0 for webcam
    parser.add_argument('--img-size', nargs= '+', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')        
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')    
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')    
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')    
    parser.add_argument('--kpt-label', action='store_true', help='use keypoint labels')
    parser.add_argument('--angle', type=float, default=80, help='Squat Angle')
    opt = parser.parse_args()
    print(opt)    

    # Check Image
    img_formats = ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
    source_file = opt.source
    image_type = source_file.split('.')[-1].lower() in img_formats

    if(image_type):
        print("Input must be video or webcam")        

    else:
        with torch.no_grad():
            detect(opt=opt)