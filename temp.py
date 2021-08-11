# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2

try:
    MAX_CORNERS = 50
    BLOCK_SIZE = 3
    QUALITY_LEVEL = 0.01
    MIN_DISTANCE = 20.0
    
    img = cv2.imread('./img/Lena.webp')
    
    if img is None:
        print('ファイルが読み込めまへん')
        import sys
        sys.exit()
        
    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, MAX_CORNERS, QUALITTY_LEVEL,
                  MIN_DISTANCE, blockSize = BLOCK_SIZE, useHarrisDetector = False)
    
    for i in corners:
        x,y = i.ravel()
        cv2.circle(img, (x,y), 4, (255, 255, 0), 2)
        
    cv2.imwrite("/Users/miyagawatakuya/.spyder-py3/OpenCV_Test/output/Lena.jpg", img)
    cv2.imshow('img', img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except:
   import sys
   print("Error:", sys.exc_info()[0])    
   print(sys.exc_info()[1])
   import traceback
   print(traceback.format_tb(sys.exc_info()[2]))
