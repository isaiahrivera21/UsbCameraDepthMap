# https://learnopencv.com/making-a-low-cost-stereo-camera-using-opencv/

import cv2 
import numpy as np 
from tqdm import tqdm

class tuneCam():
    def __init__(self,pathL,pathR):
        self.pathL = pathL
        self.pathR = pathR 
        self.criteriaStereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


    #chessBH --> #Boxes Down 
    #chessBW --> #Boxes Across 
    def Calibration_Seperate(self,chessBW,chessBH):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # tune the Left and Right cameras seperatly 
        objp = np.zeros(((chessBW-1)*(chessBH-1),3),np.float32)
        objp[:,:2] = np.mgrid[0:(chessBW-1),0:(chessBH-1)].T.reshape(-1,2)

        imgPtsL = []
        imgPtsR = []
        objPts = []

        # 1 to 12 are number of images in the folder 
        for i in tqdm(range(1,10)):
            imgL = cv2.imread(self.pathL+"left_%d.png"%i)
            imgR = cv2.imread(self.pathR+"right_%d.png"%i)

            

            imgL_gray = cv2.imread(self.pathL+"left_%d.png"%i,0)
            imgR_gray = cv2.imread(self.pathR+"right_%d.png"%i,0)

            outputL = imgL.copy()
            outputR = imgR.copy()

            retR, cornersR = cv2.findChessboardCorners(outputR,(chessBW-1,chessBH-1),None)
            retL, cornersL = cv2.findChessboardCorners(outputL,(chessBW-1,chessBH-1),None)
            # breakpoint()

            if retR and retL:
                objPts.append(objp)

                # Display stuff 
                print("HUH")
                cv2.cornerSubPix(imgR_gray,cornersR,(11,11),(-1,-1),criteria)
                cv2.cornerSubPix(imgL_gray,cornersL,(11,11),(-1,-1),criteria)
                cv2.drawChessboardCorners(outputR,((chessBW-1),(chessBH-1)),cornersR,retR)
                cv2.drawChessboardCorners(outputL,((chessBW-1),(chessBH-1)),cornersL,retL)
                cv2.imshow('cornersR',outputR)
                cv2.imshow('cornersL',outputL)
                cv2.waitKey(0)
            
                imgPtsL.append(cornersL)
                imgPtsR.append(cornersR)

            
            #Calibration 
            retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objPts,imgPtsL,imgL_gray.shape[::-1],None,None)
            hL,wL = imgL_gray.shape[:2]
            newMtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL,distL,(wL,hL),1,(wL,hL))

            retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objPts,imgPtsR,imgL_gray.shape[::-1],None,None)
            hR,wR = imgR_gray.shape[:2]
            newMtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR,distR,(wR,hR),1,(wR,hR))

            flags = 0
            flags |= cv2.CALIB_FIX_INTRINSIC

            retS, newMtxL, distL,newMtxR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(
                objPts,
                imgPtsL,
                imgPtsR,
                newMtxL,
                distL,
                newMtxR,
                distR,
                imgL_gray.shape[::-1],
                self.criteriaStereo,
                flags)
            
            # Apply stereo rectification (put camera imale planes in the same plane)
            rectifyScale= 1

            rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR = cv2.stereoRectify(
                newMtxL,
                distL,
                newMtxR,
                distR,
                imgL_gray.shape[::-1],
                Rot,
                Trns,
                rectifyScale,
                (0,0))
            
            LeftStereoMap= cv2.initUndistortRectifyMap(
                newMtxL,
                distL,
                rect_l,
                proj_mat_l,
                imgL_gray.shape[::-1],
                cv2.CV_16SC2)
            

            RightStereoMap= cv2.initUndistortRectifyMap(
                newMtxR,
                distR,
                rect_r,
                proj_mat_r,
                imgR_gray.shape[::-1],
                cv2.CV_16SC2)
            
            return LeftStereoMap, RightStereoMap
        
    def tuneDepthMap(self,paramFilePath):
        paramFile = cv2.FileStorage(paramFilePath,cv2.FILE_STORAGE_READ)





        Left_Stereo_Map_x = paramFile.getNode("Left_Stereo_Map_x").mat()
        Left_Stereo_Map_y = paramFile.getNode("Left_Stereo_Map_y").mat()
        Right_Stereo_Map_x = paramFile.getNode("Right_Stereo_Map_x").mat()
        Right_Stereo_Map_y = paramFile.getNode("Right_Stereo_Map_y").mat()

        paramFile.release()

        def nothing(x):
            pass 

        # UI Stuff 
        cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('disp',600,600)
        cv2.createTrackbar('numDisparities','disp',1,17,nothing)
        cv2.createTrackbar('blockSize','disp',5,50,nothing)
        cv2.createTrackbar('preFilterType','disp',1,1,nothing)
        cv2.createTrackbar('preFilterSize','disp',2,25,nothing)
        cv2.createTrackbar('preFilterCap','disp',5,62,nothing)
        cv2.createTrackbar('textureThreshold','disp',10,100,nothing)
        cv2.createTrackbar('uniquenessRatio','disp',15,100,nothing)
        cv2.createTrackbar('speckleRange','disp',0,100,nothing)
        cv2.createTrackbar('speckleWindowSize','disp',3,25,nothing)
        cv2.createTrackbar('disp12MaxDiff','disp',5,25,nothing)
        cv2.createTrackbar('minDisparity','disp',5,25,nothing)

        stereo = cv2.StereoBM_create()
        cam = cv2.VideoCapture(2)

        while True:

            ret, frame = cam.read()
            if not ret:
                print("FAILURE")
                break

            height, width, _ = frame.shape
            camL = frame[:, :width // 2, :]
            camR = frame[:, width // 2:, :]


            # retL, imgL = cameraL.read()
            # retR, imgR = cameraR.read()

            if ret:
                # imgR_gray = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
                # imgL_gray = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
                imgL_gray = cv2.cvtColor(camL, cv2.COLOR_BGR2GRAY)
                imgR_gray = cv2.cvtColor(camR, cv2.COLOR_BGR2GRAY)

                LeftMap = cv2.remap(
                    imgL_gray,
                    Left_Stereo_Map_x,
                    Left_Stereo_Map_y,
                    cv2.INTER_LANCZOS4,
                    cv2.BORDER_CONSTANT,
                    0)
                
                RightMap = cv2.remap(
                    imgR_gray,
                    Right_Stereo_Map_x,
                    Right_Stereo_Map_y,
                    cv2.INTER_LANCZOS4,
                    cv2.BORDER_CONSTANT,
                    0)
                
                # Updating Cam Params 
                numDisparities = cv2.getTrackbarPos('numDisparities','disp')*16
                blockSize = cv2.getTrackbarPos('blockSize','disp')*2 + 5
                preFilterType = cv2.getTrackbarPos('preFilterType','disp')
                preFilterSize = cv2.getTrackbarPos('preFilterSize','disp')*2 + 5
                preFilterCap = cv2.getTrackbarPos('preFilterCap','disp')
                textureThreshold = cv2.getTrackbarPos('textureThreshold','disp')
                uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','disp')
                speckleRange = cv2.getTrackbarPos('speckleRange','disp')
                speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','disp')*2
                disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','disp')
                minDisparity = cv2.getTrackbarPos('minDisparity','disp')

                stereo.setNumDisparities(numDisparities)
                stereo.setBlockSize(blockSize)
                stereo.setPreFilterType(preFilterType)
                stereo.setPreFilterSize(preFilterSize)
                stereo.setPreFilterCap(preFilterCap)
                stereo.setTextureThreshold(textureThreshold)
                stereo.setUniquenessRatio(uniquenessRatio)
                stereo.setSpeckleRange(speckleRange)
                stereo.setSpeckleWindowSize(speckleWindowSize)
                stereo.setDisp12MaxDiff(disp12MaxDiff)
                stereo.setMinDisparity(minDisparity)
                

                disparityMap = stereo.compute(LeftMap,RightMap)
                disparityMap = disparityMap.astype(np.float32)
                disparityMap = (disparityMap/16.0 - minDisparity)/numDisparities
                cv2.imshow("disp",disparityMap)

                if cv2.waitKey(1) == 27:
                    break 
            else:
                cam = cv2.VideoCapture(2)








                
            

            





        






def main():
    pathL = "./StereoImages/Left/"
    pathR = "./StereoImages/Right/"

    camCalibrate = tuneCam(pathL,pathR)
    paramPath = "./improved_params2.xml"
    # LeftStereoMap, RightStereoMap = camCalibrate.Calibration_Seperate(10,7)

    # print("Saving paraeters ......")
    # cv_file = cv2.FileStorage("improved_params2.xml", cv2.FILE_STORAGE_WRITE)
    # cv_file.write("Left_Stereo_Map_x",LeftStereoMap[0])
    # cv_file.write("Left_Stereo_Map_y",LeftStereoMap[1])
    # cv_file.write("Right_Stereo_Map_x",RightStereoMap[0])
    # cv_file.write("Right_Stereo_Map_y",RightStereoMap[1])
    # cv_file.release()

    camCalibrate.tuneDepthMap(paramPath)







if __name__ == "__main__":
    main()
            



# Little Note: so it seems it only made an xml file based on the first image? Might be a problem point if depth map turns out bad 


            

                
            

            
            


            
            








        




