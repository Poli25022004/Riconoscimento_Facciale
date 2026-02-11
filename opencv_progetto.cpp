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
#include <opencv2/opencv.hpp>
#include <iostream>

    int main() {

        cv::VideoCapture cap(0);
        if (!cap.isOpened()) return -1;

        cv::Ptr<cv::BackgroundSubtractor> bgSub =
            cv::createBackgroundSubtractorMOG2();

        cv::KalmanFilter kf(4, 2);
        kf.transitionMatrix = (cv::Mat_<float>(4, 4) <<
            1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1);

        kf.measurementMatrix = cv::Mat::eye(2, 4, CV_32F);
        kf.processNoiseCov = cv::Mat::eye(4, 4, CV_32F) * 1e-2;
        kf.measurementNoiseCov = cv::Mat::eye(2, 2, CV_32F) * 1e-1;
        kf.errorCovPost = cv::Mat::eye(4, 4, CV_32F);

        cv::Mat frame, fgMask;

        while (true) {
            cap >> frame;
            if (frame.empty()) break;

            bgSub->apply(frame, fgMask);

            // Pulizia rumore
            cv::erode(fgMask, fgMask, cv::Mat(), cv::Point(-1, -1), 2);
            cv::dilate(fgMask, fgMask, cv::Mat(), cv::Point(-1, -1), 2);

            // Troviamo contorni
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(fgMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            if (!contours.empty()) {
                // Troviamo il blob più grande
                int maxIdx = 0;
                double maxArea = 0;

                for (int i = 0; i < contours.size(); i++) {
                    double area = cv::contourArea(contours[i]);
                    if (area > maxArea) {
                        maxArea = area;
                        maxIdx = i;
                    }
                }

                cv::Rect bound = cv::boundingRect(contours[maxIdx]);
                float centerX = bound.x + bound.width / 2;
                float centerY = bound.y + bound.height / 2;

                // Misura per Kalman
                cv::Mat measurement(2, 1, CV_32F);
                measurement.at<float>(0) = centerX;
                measurement.at<float>(1) = centerY;

                kf.correct(measurement);
            }

            // Predizione
            cv::Mat prediction = kf.predict();
            float predX = prediction.at<float>(0);
            float predY = prediction.at<float>(1);

            // Disegno
            cv::circle(frame, cv::Point(predX, predY), 20, cv::Scalar(0, 255, 0), 3);

            cv::imshow("Tracking Movimento", frame);
            cv::imshow("Mask", fgMask);

            if (cv::waitKey(30) == 27) break;
        }

        return 0;
    }

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

        // Predizione Kalman
        cv::Mat prediction = kf.predict();
        float pred_x = prediction.at<float>(0);
        float pred_y = prediction.at<float>(1);

        // Disegniamo il rettangolo predetto
        cv::rectangle(frame,
            cv::Point(pred_x - 50, pred_y - 50),
            cv::Point(pred_x + 50, pred_y + 50),
            cv::Scalar(0, 255, 0), 2
        );

        cv::imshow("Face Tracking", frame);

        if (cv::waitKey(30) == 27) break; // ESC per uscire
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
