import cv2 as cv
import numpy as np
from skimage.metrics import mean_squared_error
import csv

def create_results(TYPE):

    # with open('csvs/'+TYPE+'_mse.csv', 'w', newline='') as csvfile:
    #         fieldnames = ['MSE']
    #         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #         writer.writeheader()

    indices = ["{0:03}".format(i) for i in range(120)]

    for i in indices:
        img1 = cv.imread('results/'+TYPE+'/google-vision/flythrough/seged_img/'+i+'.png')
        img2 = cv.imread('results/'+TYPE+'/grounding-dino/flythrough/seged_img/'+i+'.png')

        img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

        mse = mean_squared_error(img1_gray, img2_gray)

        print("MSE value:", mse)        

        diff = np.abs(img1_gray - img2_gray) 

        cv.imwrite("mse_images/"+TYPE+"/"+i+".png", diff)

        with open('csvs/'+TYPE+'_mse.csv', 'a', newline='') as csvfile:
            csvfile.write('\n'+str(mse))

if __name__ == "__main__":
    TYPE="flower"
    create_results(TYPE)