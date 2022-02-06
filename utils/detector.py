import cv2
import numpy as np
from skimage.measure import LineModelND, ransac

class Detector():
    def __init__(self, scan_range={'start':0,'stop':240,'steps':20},scan_window={'height':15,'max_adjust':10}):
        self.scan_range = scan_range
        self.scan_window = scan_window
        self.model = LineModelND()
        self.debug = []
    
    def img_filter(self,iimg):
        """
        Filters an RGB image to get the lane boundaries

        Args:
            iimg ([type]): RGB image

        Returns:
            [type]: RGB image
        """
        iimg = cv2.cvtColor(iimg,cv2.COLOR_BGR2GRAY)
        ciimg = cv2.Canny(iimg,50,250)
        ciimg = cv2.morphologyEx(ciimg,cv2.MORPH_DILATE,(1,1),iterations = 5)
    
        siimg = cv2.Sobel(iimg,cv2.CV_8U,1,0,ksize=1)
        siimg = cv2.morphologyEx(siimg,cv2.MORPH_DILATE,(1,1),iterations = 5)
        ret,siimg= cv2.threshold(siimg,50,255,cv2.THRESH_OTSU)
        siimg = cv2.morphologyEx(siimg, cv2.MORPH_CLOSE, (3,3))
        
        iimg= cv2.bitwise_and(ciimg,ciimg,mask=siimg)
        
        return iimg
        
    def get_lane_detections(self,iimg,start={'x':105, 'y':230},stop={'x':135, 'y':230},label='mid', use_RANSAC=True, debug=False):
        """
        Parses the input image, with virtual sensors, detects the peeks and save the points

        Args:
            iimg ([type]): 1 channel gray image
            start (dict, optional): detection area start. Defaults to {'x':105, 'y':230}.
            stop (dict, optional): detection area start. Defaults to {'x':135, 'y':230}.
            label (str, optional): name of the line. Defaults to 'mid'.
            use_RANSAC (bool, optional): Use RANSAC. Defaults to True.
            debug (bool, optional): collect debug info. Defaults to False.

        Returns:
            [type]: detections coordinates 
        """
            
        adjust = 0
        minx = min(start['x'],stop['x'])
        maxx = max(start['x'],stop['x']) + adjust
        detections = []
        for i in range (self.scan_range['start'],self.scan_range['stop'],self.scan_range['steps']):
            # detections y coordinate
            y = start['y']-i

            # get detections from a line
            det_line = iimg[y:y+self.scan_window['height'], minx:maxx]
            
            # scan an image segment, sum detection
            hist = np.sum(det_line, axis=0)
            
            # get peek location
            peek = np.argmax(hist)
            
            # define threshold = average, find peeks, 
            # update 'sensor location'= peek centered
            if hist[peek] > np.average(hist):
                x1 = minx+peek
                y1 = y
                det_mid_x = minx+len(hist)//2
                
                adjust = x1 - det_mid_x
                
                # apply adjust only if in defined range
                if np.abs(adjust) >= self.scan_window['max_adjust']:
                    sing = np.sign(adjust)
                    adjust = sing * self.scan_window['max_adjust']
                
                minx += adjust
                maxx += adjust
                
                #self.buffer.append({label:[x1,y1]})
                detections.append([x1,y1])
                if debug == True:
                    self.debug.append({'detection':[x1,y1], 'detection_mid':[det_mid_x,y1],'rectangle': [minx,y1,minx+len(hist),y1+self.scan_window['height']]}) 
        
        if use_RANSAC == True:
            _, inliers = self.filter_outliers(detections)
            if inliers is not None:
                detections = np.array(detections)[inliers]

        return {label:detections}
                      
    def filter_outliers(self,data):
        """
        apply RUNSAC

        Args:
            data ([type]): data points

        Returns:
            [type]: filtered list
        """
        model_robust = None
        inliers = None
        
        data = np.array(data)
        self.model.estimate(data)
        try:
            model_robust, inliers = ransac(data, LineModelND, min_samples=2, residual_threshold=1, max_trials=200)
        except:
            pass
        
        return model_robust,inliers
        
    def draw_detections(self, iimg,data):
        """
        Visualize detections

        Args:
            iimg ([type]): RGB image
            data ([type]): points detected

        Returns:
            [type]: RGB image
        """
        limg = iimg.copy()
        for v in data:
            cv2.circle(limg,(v[0],v[1]),2,[255],-1)
        if len(self.debug)>0:
            for v in self.debug:
                # window center
                cv2.circle(limg, (v['detection_mid'][0],v['detection_mid'][1]+5),1,[255])
                # detection rectangle
                cv2.rectangle(limg,(v['rectangle'][0],v['rectangle'][1]),(v['rectangle'][2],v['rectangle'][3]),[255],1)
        
        return limg
