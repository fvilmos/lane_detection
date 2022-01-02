import numpy as np

class Interpolator():
    
    def __init__(self,max_poly_degree=3):
        self.max_poly_degree=max_poly_degree
        
    def interpolate(self,indata=dict(), ip_params={'start':0, 'stop':240, 'steps':20},key='mid', nr_interpolated_points=240, equ_selector=False, debug=False):
        """
        Takes detected points, find the corresponding polynom fits the data.

        Args:
            indata ([type], optional): detected points. Defaults to dict().
            ip_params (dict, optional): interpolation info. Defaults to {'start':0, 'stop':240, 'steps':20}.
            key (str, optional): Name of the line. Defaults to 'mid'.
            nr_interpolated_points (int, optional): Max number of interpolation points. Defaults to 240.
            equ_selector (bool, optional): search the best fitting equation (line / curve, etc.). Defaults to False.
            debug (bool, optional): Collects debug info. Defaults to False.

        Returns:
            [type]: interpolated points
        """
        data = np.array(indata[0][key])

        x_coord = data[:,0]
        y_coord = data[:,1]
        
        min_mse_pos = self.max_poly_degree
        
        # polynomial degree selector
        if equ_selector == True:
            # find the best fit
            best_poly = []
            for i in range(0,self.max_poly_degree):
                pfit = np.polyfit(y_coord,x_coord,i)
                polynom = np.poly1d(pfit)
                test_y = polynom(x_coord)
                difference= y_coord-test_y
                st_d = np.std(difference)                
                best_poly.append(st_d)

                if debug==True:
                    print ("polynom: ",polynom, "mse: ",mse)

            # select best poly
            min_mse_pos = np.argmin(np.array(best_poly))

        # order start from 1, position from 0
        pfit = np.polyfit(y_coord,x_coord,min_mse_pos)
        polynom = np.poly1d(pfit)
        
        y_ipp = np.float32(np.linspace(ip_params['start'],ip_params['stop'],ip_params['steps']))
        x_ipp = polynom(y_ipp)

        ply_coords = np.column_stack((x_ipp,y_ipp))
        return {key:ply_coords}
    
    def echidistant_lane(self, lane_pts, distnce=40, return_end_point=True,lane_side=1):
        """
        Generates an equidistant curve based on the perpendicular ponts of the givin curve and disctance

        Args:
            lane_pts ([type]): detected points
            distnce (int, optional): distance from the input curve. Defaults to 40.
            return_end_point (bool, optional): Generate perpendicular line or just end point. Defaults to True.
            lane_side (int, optional): [description]. Defaults to 1.

        Returns:
            [type]: equidistant curve points
        """
        buffer = []
        for i in range (0, len(lane_pts[0])-1):
            # compute m1 = dy/dx
            p0 = lane_pts[0][i]
            p1 = lane_pts[0][i+1]

            dx = p1[0]-p0[0]
            dy = p1[1]-p0[1]
            m1 = dy/dx

            # compute m2 = -1/m1
            # line y = m2 * x + b, b=0
            m2 = -1.0/m1
            x = np.linspace(0,distnce,10)
            y = m2*x

            nx = p1[0]-x*lane_side
            ny = p1[1]-y*lane_side

            if return_end_point == True:
                nx = nx[-1]
                ny = ny[-1]
            norm_pts = np.int32(np.column_stack((nx,ny)))
            buffer.append(norm_pts)

        return np.array(buffer, dtype=np.int32)
