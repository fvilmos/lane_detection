import cv2
import numpy as np

class ImageWarp():
    
    def __init__(self,img_h=240, img_w=320, offset=150, src=[[50, 240], [200, 240], [0, 0], [320, 0]],dst=[[135, 240], [150, 240], [0, 0], [320, 0]]):
        self.img_h = img_h
        self.img_w = img_w
        self.warp_offset = offset
        self.src = np.float32(src)
        self.dst = np.float32(dst)
        self.wmat = cv2.getPerspectiveTransform(self.src, self.dst)
        self.wmat_inv = cv2.getPerspectiveTransform(self.dst,self.src)
    
    def img_warp(self, iimg,inv=False, offset=False):
        """
        Warps an image based on the input parameters

        Args:
            iimg ([type]): RGB / Gray image
            inv (bool, optional): invers transformation. Defaults to False.
            offset (bool, optional): use offset for warping the image. Defaults to False.

        Returns:
            [type]: warped image
        """
        
        ret = []
        timg = None
        
        if offset == True:
            timg = iimg[self.warp_offset:self.warp_offset+self.img_h, 0:self.img_w]
        else:
            timg = iimg
        
        if inv == False:
            ret = cv2.warpPerspective(timg,self.wmat , (self.img_w, self.img_h))
        else:
            ret = cv2.warpPerspective(timg,self.wmat_inv , (self.img_w, self.img_h))
        return ret
    
    def pts_unwarp(self,pts):
        """
        Backprojects points from warped image to un-warped image

        Args:
            pts ([type]): points to backproject

        Returns:
            [type]: backprojected points
        """
        return cv2.perspectiveTransform (pts, self.wmat_inv)