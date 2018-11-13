import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import cv2
from calibrate_camera import calibrateCamera


from LaneFinder import LaneFinder


mtx, dist = calibrateCamera()
lane_finder = LaneFinder(mtx, dist)

# for videoName in os.listdir("test_videos"):
#     print(videoName)
#     output_name = 'output_videos/' + videoName
#     filename = 'test_videos/'+videoName
#     clip = VideoFileClip(filename)
#     processed_clip = clip.fl_image(lane_finder.run)
#     processed_clip.write_videofile(output_name, audio=False)

for imageName in os.listdir("test_images"):
    img = cv2.imread('test_images/'+imageName)
    cv2.imshow(imageName, lane_finder.run(img))
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('output_images/'+imageName, img)
        cv2.destroyAllWindows()

#     plt.imshow(lane_finder.run(img))
#     plt.show()

#     y_eval = np.max(self.ploty)
#     left_curverad = ((1 + (2*self.left_fitx[0]*y_eval + self.left_fitx[1])**2)**1.5) / np.absolute(2*self.left_fitx[0])
#     right_curverad = ((1 + (2*self.right_fitx[0]*y_eval + self.right_fitx[1])**2)**1.5) / np.absolute(2*self.right_fitx[0])

#     # Plots the left and right polynomials on the lane lines
#     plt.plot(left_fitx, ploty, color='red')
#     plt.plot(right_fitx, ploty, color='red')


