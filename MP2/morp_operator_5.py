import cv2 as cv
import numpy as np


def create_SE(dimensions):
    return np.ones((dimensions[0], dimensions[1]), np.uint8)*255

def create_SE_cross(dimensions):
    SE = np.zeros(dimensions, np.uint8)
    center = (dimensions[0] // 2, dimensions[1] // 2)
    SE[:, center[1]] = 255
    SE[center[0], :] = 255
    return SE

def dilate(image, struct_elem):
    height, width = image.shape
    se_h, se_w = struct_elem.shape
    padding_h, padding_w = se_h // 2, se_w // 2
    
    # Create an output image, initialized with zeros (black)
    output = np.zeros((height, width), dtype=np.uint8)

    for i in range(padding_h, height - padding_h):
        for j in range(padding_w, width - padding_w):
            # If there's a '1' in the structuring element's neighborhood, set the output pixel to '1'
            region = image[i - padding_h:i + padding_h + 1, j - padding_w:j + padding_w + 1]
            output[i, j] = np.max(region & struct_elem)

    return output

def erode(image, struct_elem):
    height, width = image.shape
    se_h, se_w = struct_elem.shape
    padding_h, padding_w = se_h // 2, se_w // 2
    
    # Create an output image, initialized with zeros (black)
    output = np.zeros((height, width), dtype=np.uint8)

    for i in range(padding_h, height - padding_h):
        for j in range(padding_w, width - padding_w):
            # If all '1's in the structuring element match the image, set the output pixel to '1'
            region = image[i - padding_h:i + padding_h + 1, j - padding_w:j + padding_w + 1]
            output[i, j] = np.min(region | (~struct_elem))

    return output

def Opening(img, SE):
    return dilate(erode(img,SE),SE)

def Closing(img, SE):
    return erode(dilate(img,SE),SE)

def Boundary(img, SE):
    output= dilate(img,SE)
    return output - erode(output, SE)


def main():
    # GUN image
    gun_img = cv.imread('images/gun.bmp', cv.IMREAD_GRAYSCALE)
    #create SE
    gun_SE_3 = create_SE([3, 3])
    gun_SE_5 = create_SE([5, 5])
    gun_SE_7 = create_SE([7, 7])
    gun_SE_cross = create_SE_cross((3, 3))
    
    # #Dilation
    gun_dialte_1= dilate(gun_img,gun_SE_3)
    cv.imshow('gun dialte 3',gun_dialte_1)
    cv.imwrite('gun dialte 3.bmp', gun_dialte_1)
    
    gun_dialte_2= dilate(gun_img,gun_SE_5)
    cv.imshow('gun dialte 5',gun_dialte_2)
    cv.imwrite('gun dialte 5.bmp', gun_dialte_2)

    gun_dialte_3= dilate(gun_img,gun_SE_7)
    cv.imshow('gun dialte 7',gun_dialte_3)
    cv.imwrite('gun dialte 7.bmp', gun_dialte_3)
   
    gun_dialte_4= dilate(gun_img,gun_SE_cross)
    cv.imshow('gun dialte cross',gun_dialte_4)
    cv.imwrite('gun dialte cross.bmp', gun_dialte_4)
    cv.waitKey(0)    

    #Erosion
    gun_erode_1= erode(gun_img,gun_SE_3)
    cv.imshow('gun erode 3',gun_erode_1)
    cv.imwrite('gun erode 3.bmp', gun_erode_1)
    
    gun_erode_2= erode(gun_img,gun_SE_5)
    cv.imshow('gun erode 5',gun_erode_2)
    cv.imwrite('gun erode 5.bmp', gun_erode_2)

    gun_erode_3= erode(gun_img,gun_SE_7)
    cv.imshow('gun erode 7',gun_erode_3)
    cv.imwrite('gun erode 7.bmp', gun_erode_3)
   
    gun_erode_4= erode(gun_img,gun_SE_cross)
    cv.imshow('gun erode cross',gun_erode_4)
    cv.imwrite('gun erode cross.bmp', gun_erode_4)
    cv.waitKey(0)

    # Opening
    gun_Opening_1= Opening(gun_img,gun_SE_3)
    cv.imshow('gun Opening 3',gun_Opening_1)
    cv.imwrite('gun Opening 3.bmp', gun_Opening_1)
    
    gun_Opening_2= Opening(gun_img,gun_SE_5)
    cv.imshow('gun Opening 5',gun_Opening_2)
    cv.imwrite('gun Opening 5.bmp', gun_Opening_2)

    gun_Opening_3= Opening(gun_img,gun_SE_7)
    cv.imshow('gun Opening 7',gun_Opening_3)
    cv.imwrite('gun Opening 7.bmp', gun_Opening_3)
   
    gun_Opening_4= Opening(gun_img,gun_SE_cross)
    cv.imshow('gun Opening cross',gun_Opening_4)
    cv.imwrite('gun Opening cross.bmp', gun_Opening_4)
    cv.waitKey(0)  

    # Closing
    gun_Closing_1= Closing(gun_img,gun_SE_3)
    cv.imshow('gun Closing 3',gun_Closing_1)
    cv.imwrite('gun Closing 3.bmp', gun_Closing_1)
    
    gun_Closing_2= Closing(gun_img,gun_SE_5)
    cv.imshow('gun Closing 5',gun_Closing_2)
    cv.imwrite('gun Closing 5.bmp', gun_Closing_2)

    gun_Closing_3= Closing(gun_img,gun_SE_7)
    cv.imshow('gun Closing 7',gun_Closing_3)
    cv.imwrite('gun Closing 7.bmp', gun_Closing_3)
   
    gun_Closing_4= Closing(gun_img,gun_SE_cross)
    cv.imshow('gun Closing cross',gun_Closing_4)
    cv.imwrite('gun Closing cross.bmp', gun_Closing_4)
    cv.waitKey(0) 

    # Boundary
    gun_Boundary_1= Boundary(gun_img,gun_SE_3)
    cv.imshow('gun Boundary 3',gun_Boundary_1)
    cv.imwrite('gun Boundary 3.bmp', gun_Boundary_1)
    
    gun_Boundary_2= Boundary(gun_img,gun_SE_5)
    cv.imshow('gun Boundary 5',gun_Boundary_2)
    cv.imwrite('gun Boundary 5.bmp', gun_Boundary_2)

    gun_Boundary_3= Boundary(gun_img,gun_SE_7)
    cv.imshow('gun Boundary 7',gun_Boundary_3)
    cv.imwrite('gun Boundary 7.bmp', gun_Boundary_3)
   
    gun_Boundary_4= Boundary(gun_img,gun_SE_cross)
    cv.imshow('gun Boundary cross',gun_Boundary_4)
    cv.imwrite('gun Boundary cross.bmp', gun_Boundary_4)
    cv.waitKey(0) 

    # PALM image
    palm_img = cv.imread('images/palm.bmp', cv.IMREAD_GRAYSCALE)
    #create SE
    palm_SE_3 = create_SE([3, 3])
    palm_SE_5 = create_SE([5, 5])
    palm_SE_7 = create_SE([7, 7])
    palm_SE_cross = create_SE_cross((3, 3))
    
    # #Dilation
    palm_dialte_1= dilate(palm_img,palm_SE_3)
    cv.imshow('palm dialte 3',palm_dialte_1)
    cv.imwrite('palm dialte 3.bmp', palm_dialte_1)
    
    palm_dialte_2= dilate(palm_img,palm_SE_5)
    cv.imshow('palm dialte 5',palm_dialte_2)
    cv.imwrite('palm dialte 5.bmp', palm_dialte_2)

    palm_dialte_3= dilate(palm_img,palm_SE_7)
    cv.imshow('palm dialte 7',palm_dialte_3)
    cv.imwrite('palm dialte 7.bmp', palm_dialte_3)
   
    palm_dialte_4= dilate(palm_img,palm_SE_cross)
    cv.imshow('palm dialte cross',palm_dialte_4)
    cv.imwrite('palm dialte cross.bmp', palm_dialte_4)
    cv.waitKey(0)    

    # #Erosion
    palm_erode_1= erode(palm_img,palm_SE_3)
    cv.imshow('palm erode 3',palm_erode_1)
    cv.imwrite('palm erode 3.bmp', palm_erode_1)
    
    palm_erode_2= erode(palm_img,palm_SE_5)
    cv.imshow('palm erode 5',palm_erode_2)
    cv.imwrite('palm erode 5.bmp', palm_erode_2)

    palm_erode_3= erode(palm_img,palm_SE_7)
    cv.imshow('palm erode 7',palm_erode_3)
    cv.imwrite('palm erode 7.bmp', palm_erode_3)
   
    palm_erode_4= erode(palm_img,palm_SE_cross)
    cv.imshow('palm erode cross',palm_erode_4)
    cv.imwrite('palm erode cross.bmp', palm_erode_4)
    cv.waitKey(0)

    # Opening
    palm_Opening_1= Opening(palm_img,palm_SE_3)
    cv.imshow('palm Opening 3',palm_Opening_1)
    cv.imwrite('palm Opening 3.bmp', palm_Opening_1)
    
    palm_Opening_2= Opening(palm_img,palm_SE_5)
    cv.imshow('palm Opening 5',palm_Opening_2)
    cv.imwrite('palm Opening 5.bmp', palm_Opening_2)

    palm_Opening_3= Opening(palm_img,palm_SE_7)
    cv.imshow('palm Opening 7',palm_Opening_3)
    cv.imwrite('palm Opening 7.bmp', palm_Opening_3)
   
    palm_Opening_4= Opening(palm_img,palm_SE_cross)
    cv.imshow('palm Opening cross',palm_Opening_4)
    cv.imwrite('palm Opening cross.bmp', palm_Opening_4)
    cv.waitKey(0)  

    # Closing
    palm_Closing_1= Closing(palm_img,palm_SE_3)
    cv.imshow('palm Closing 3',palm_Closing_1)
    cv.imwrite('palm Closing 3.bmp', palm_Closing_1)
    
    palm_Closing_2= Closing(palm_img,palm_SE_5)
    cv.imshow('palm Closing 5',palm_Closing_2)
    cv.imwrite('palm Closing 5.bmp', palm_Closing_2)

    palm_Closing_3= Closing(palm_img,palm_SE_7)
    cv.imshow('palm Closing 7',palm_Closing_3)
    cv.imwrite('palm Closing 7.bmp', palm_Closing_3)
   
    palm_Closing_4= Closing(palm_img,palm_SE_cross)
    cv.imshow('palm Closing cross',palm_Closing_4)
    cv.imwrite('palm Closing cross.bmp', palm_Closing_4)
    cv.waitKey(0) 

    # Boundary
    palm_Boundary_1= Boundary(palm_img,palm_SE_3)
    cv.imshow('palm Boundary 3',palm_Boundary_1)
    cv.imwrite('palm Boundary 3.bmp', palm_Boundary_1)
    
    palm_Boundary_2= Boundary(palm_img,palm_SE_5)
    cv.imshow('palm Boundary 5',palm_Boundary_2)
    cv.imwrite('palm Boundary 5.bmp', palm_Boundary_2)

    palm_Boundary_3= Boundary(palm_img,palm_SE_7)
    cv.imshow('palm Boundary 7',palm_Boundary_3)
    cv.imwrite('palm Boundary 7.bmp', palm_Boundary_3)
   
    palm_Boundary_4= Boundary(palm_img,palm_SE_cross)
    cv.imshow('palm Boundary cross',palm_Boundary_4)
    cv.imwrite('palm Boundary cross.bmp', palm_Boundary_4)
    cv.waitKey(0) 


if __name__ == '__main__':
    main()