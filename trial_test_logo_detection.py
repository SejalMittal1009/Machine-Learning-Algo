from imageai.Detection.Custom import CustomObjectDetection
import numpy as np
detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("D://Ansh//Documents//SEM-7//DV//Project//data//models//detection_model-ex-005--loss-0005.176.h5")
detector.setJsonPath("D://Ansh//Documents//SEM-7//DV//Project//data//json//detection_config.json")
detector.loadModel()
#temp=['image1028.xml', 'image1032.xml', 'image1042.xml', 'image1044.xml', 'image1072.xml', 'image1074.xml', 'image1076.xml', 'image1085.xml', 'image1090.xml', 'image1372.xml', 'image1373.xml', 'image1374.xml', 'image1375.xml', 'image1376.xml', 'image1377.xml', 'image1394.xml', 'image1395.xml', 'image1396.xml', 'image1398.xml', 'image1399.xml', 'image1400.xml', 'image1404.xml', 'image1405.xml', 'image1413.xml', 'image655.xml', 'image656.xml', 'image657.xml', 'image658.xml', 'image659.xml']
#temp=[i[:-3]+"jpg" for i in temp]
temp=["chevrolet.jpg","hyundai.jpg","maruti.jpg","toyota.jpg"]
#temp=["maruti.jpg"]
path="D://Ansh//Documents//SEM-7//DV//Project//"

for i in temp:
    temp_ct={"Chevrolet":0,"Hyundai":0,"Maruti":0,"Toyota":0}
    temp_acc={"Chevrolet":0,"Hyundai":0,"Maruti":0,"Toyota":0}
    temp_coordi={"Chevrolet":np.array([0,0,0,0]),"Hyundai":np.array([0,0,0,0]),"Maruti":np.array([0,0,0,0]),"Toyota":np.array([0,0,0,0])}
    detections = detector.detectObjectsFromImage(input_image=path+i, output_image_path=path+"logo_detected_"+i,minimum_percentage_probability=30)
    for detection in detections:
        temp_ct[detection["name"]]+=1
        temp_acc[detection["name"]]+=detection["percentage_probability"]
        temp_coordi[detection["name"]]+=np.array(detection["box_points"])
        print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
    for key in temp_ct:
        if temp_ct[key]!=0:
            temp_acc[key]=temp_acc[key]/temp_ct[key]
            temp_coordi[key]=temp_coordi[key]/temp_ct[key]
    for key in sorted(temp_acc,key=temp_acc.get,reverse=True):
        print(key,temp_acc[key],list(temp_coordi[key]))
