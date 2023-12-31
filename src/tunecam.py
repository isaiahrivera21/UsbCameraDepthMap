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



def main():
    pathL = "./StereoImages/Left/"
    pathR = "./StereoImages/Right/"

    camCalibrate = tuneCam(pathL,pathR)
    LeftStereoMap, RightStereoMap = camCalibrate.Calibration_Seperate(10,7)

    print("Saving paraeters ......")
    cv_file = cv2.FileStorage("improved_params2.xml", cv2.FILE_STORAGE_WRITE)
    cv_file.write("Left_Stereo_Map_x",LeftStereoMap[0])
    cv_file.write("Left_Stereo_Map_y",LeftStereoMap[1])
    cv_file.write("Right_Stereo_Map_x",RightStereoMap[0])
    cv_file.write("Right_Stereo_Map_y",RightStereoMap[1])
    cv_file.release()



if __name__ == "__main__":
    main()
            



# Little Note: so it seems it only made an xml file based on the first image? Might be a problem point if depth map turns out bad 


            

                
            

            
            


            
            








        




