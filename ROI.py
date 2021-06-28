import numpy as np
import cv2
from matplotlib import pyplot as plt

class ROI(object):
    def __init__(self):
        self.capture = cv2.VideoCapture('RickAndMorty.mkv')

        # Bounding box reference points and boolean if we are extracting coordinates
        self.image_coordinates = []
        self.extract = False
        self.selected_ROI = False
        self.grayscale = None
        self.blackAndWhite = None
        self.x1=0
        self.x2=0
        self.y1=0
        self.y2=0

        self.update()

    def update(self):
        while True:
            if self.capture.isOpened():
                # Read frame
                (self.status, self.frame) = self.capture.read()
                

                # Crop image
                self.clone = self.frame.copy()
                cv2.namedWindow('image')
                cv2.setMouseCallback('image', self.extract_coordinates)
                #cv2.imshow('image', self.clone)

                # Crop and display cropped image
                if self.selected_ROI:
                    if self.crop_ROI():
                        cv2.rectangle(self.frame, (self.x1, self.y1), (self.x2, self.y2), (255,0,0), 2)
                        self.show_cropped_ROI()
                # Close program with keyboard 'q'
                cv2.imshow('image', self.frame)
                key = cv2.waitKey(2)
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    break
            else:
                pass

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates = [(x,y)]
            self.extract = True
            self.selected_ROI = False

        # Record ending (x,y) coordintes on left mouse bottom release
        elif event == cv2.EVENT_LBUTTONUP:
            self.image_coordinates.append((x,y))
            self.extract = False
            self.selected_ROI = True

    def crop_ROI(self):
        if self.selected_ROI:
            self.cropped_image = self.frame.copy()
            
            self.x1 = self.image_coordinates[0][0]
            self.y1 = self.image_coordinates[0][1]
            self.x2 = self.image_coordinates[1][0]
            self.y2 = self.image_coordinates[1][1]
         
            if self.x1 > self.x2:
                tmp = self.x1
                self.x1 = self.x2
                self.x2 = tmp
            if self.y1 > self.y2:    
                tmp = self.y1
                self.y1 = self.y2
                self.y2 = tmp
            
            if (self.y2 - self.y1) < 1 or (self.x2 - self.x1) < 1:
                self.selected_ROI = False
                return False
            
            self.cropped_image = self.cropped_image[self.y1:self.y2, self.x1:self.x2]
            grayscale = cv2.cvtColor(self.cropped_image, cv2.COLOR_BGR2GRAY)
            (thresh, self.blackAndWhite) = cv2.threshold(grayscale, 128, 255, cv2.THRESH_BINARY)

            #print('Cropped image: {} {}'.format(self.image_coordinates[0], self.image_coordinates[1]))
        else:
            print('Select ROI to crop before cropping')
        return True

    def show_cropped_ROI(self):
        #cv2.imshow('cropped image', self.blackAndWhite)
        print("Calculate histogram")
        plt.hist(self.blackAndWhite.ravel(),256,[0,256])
        plt.show()
        plt.pause(0.0001)

if __name__ == '__main__':
    static_ROI = ROI()
