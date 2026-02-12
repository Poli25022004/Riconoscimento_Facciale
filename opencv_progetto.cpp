#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <iostream>

int main() {
    // 1. Apriamo la webcam
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Errore nell'apertura della webcam!" << std::endl;
        return -1;
    }

    // 2. Carichiamo il modello Haar Cascade per il volto
    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load("haarcascade_frontalface_default.xml")) {
        std::cerr << "Errore nel caricamento del modello Haar!" << std::endl;
        return -1;
    }

    // 3. Creiamo un filtro di Kalman
    cv::KalmanFilter kf(4, 2, 0); // stato: x,y,vx,vy; misura: x,y
    kf.transitionMatrix = (cv::Mat_<float>(4, 4) <<
        1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1
        );
    kf.measurementMatrix = cv::Mat::eye(2, 4, CV_32F);
    kf.processNoiseCov = cv::Mat::eye(4, 4, CV_32F) * 1e-2;
    kf.measurementNoiseCov = cv::Mat::eye(2, 2, CV_32F) * 1e-1;
    kf.errorCovPost = cv::Mat::eye(4, 4, CV_32F);

    cv::Mat measurement(2, 1, CV_32F);

    while (true) {
        cv::Mat frame, gray;
        cap >> frame; // cattura frame
        if (frame.empty()) break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY); // convertiamo in grayscale

        // 4. Rileviamo i volti
        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(30, 30));

        if (!faces.empty()) {
            // Prendiamo il primo volto rilevato
            cv::Rect face = faces[0];
            measurement.at<float>(0) = face.x + face.width / 2;
            measurement.at<float>(1) = face.y + face.height / 2;

            // Aggiorniamo Kalman con la misura
            kf.correct(measurement);
        }

        // 5. Predizione Kalman
        cv::Mat prediction = kf.predict();
        float pred_x = prediction.at<float>(0);
        float pred_y = prediction.at<float>(1);

        // 6. Disegniamo il rettangolo predetto
        cv::rectangle(frame,
            cv::Point(pred_x - 50, pred_y - 50),
            cv::Point(pred_x + 50, pred_y + 50),
            cv::Scalar(0, 255, 0), 2
        );

        cv::imshow("Face Tracking con Kalman", frame);

        if (cv::waitKey(30) == 27) break; // ESC per uscire
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
