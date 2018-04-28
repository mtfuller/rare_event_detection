from ucsd_dataset import ucsd_dataset
from models.c3d_model import C3DModel
from models.bc_model import BCModel
from video import Video

def main():
    # Load in one of the videos from the dataset
    folder = "Assault049_x264/"
    selected_video = Video(folder, 1)
    selected_video.resize(112,112)
    selected_video = selected_video.getSegments()

    # Run video through the pre-trained C3D model to obtain C3D features
    c3d = C3DModel()
    features, elapsed_time = c3d.predict(selected_video)
    print("Extracting features...")
    print("Elapsed time: %fs" % (elapsed_time))

    # Load the trained Binary Classifier model to predict the anomaly score of the C3D features
    bc = BCModel()
    bc.load_model("models/UCF_Train/intervals/save-00004/model")
    scores, elapsed_time = bc.predict(features)
    print("Predicting scores...")
    print("Elapsed time: %fs" % (elapsed_time))

    # Iterate through and print out each 16-frame anomaly score and
    for i in range(len(scores)):
        print("Frames (%4d to %4d)\tScore: %f\t%s" % (i*16,(i+1)*16,scores[i], "ANOMALY!!!" if scores[i] > .5 else ""))

if __name__ == '__main__':
    main()