import cv2
print(cv2.__version__)


terrainlist=['bark','dirt','dry_veg','fallen','foliage','grass','paved','sand']
lengthlist=[948,1873,4219,128,1653,1025,2313,245]
for j in range(len(terrainlist)):
    terrain=terrainlist[j]
    print(terrain)
    length=lengthlist[j]
    for i in range(length):
        print(i)
        img=cv2.imread('C:/cs229/Data/frames/original/'+terrain+'/'+terrain+' 1920 1080 ('+str(i+1)+').jpg')
        crop_img = img[420:1500, 0:1080] 
        # cv2.imshow("cropped", crop_img)
        # cv2.waitKey(0)
        
        height, width = 100,100
        res = cv2.resize(img,(width, height))
        
        cv2.imwrite("C:/cs229/Data/frames/"+str(height)+" "+str(width)+"/"+terrain+"/"+terrain+" "+str(height)+"_"+str(width)+"_"+str(i)+".jpg" , res)     # save frame as JPEG file
