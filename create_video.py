import cv2
import numpy as np



def main():
    """Takes a directory of frames and creates a .avi file using those images.
    """

    # Ask use for the name of the video and how many frames to use
    url = 'dataset/' + input("What is the name of the video: ")
    img = []
    frames = int(input("How many total frames: "))

    # Collect all the images into a Python list
    print("Gathering images...")
    for i in range(0, frames+1):
        print("Progress: %2.1f" % (float(i) / frames))
        img.append(cv2.imread(url + '/frame' + str(i) + '.jpg'))
    print("Images loaded!!!")

    height, width, layers = img[1].shape

    # Convert the list of images to a video and saves it to the working directory
    print("Creating video writer...")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter('./output.avi', fourcc, 30.0, (width, height))
    print("Video writer created!")

    print("Total Frames: %d" % (len(img)))

    # Writes each image to the video
    print("Writing Frames...")
    for j in range(0, frames+1):
        print("Progress: %2.1f" % (float(j)/frames))
        video.write(img[j])

    # Finalization
    print("Finalizing...")
    cv2.destroyAllWindows()
    video.release()
    print("Finished!!!")

if __name__ == '__main__':
    main()