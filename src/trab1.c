/* Para compilar:
 * gcc trab1.c -o trab1 `pkg-config opencv --cflags --libs`
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cv.h>
#include <highgui.h>

#define cascade_name "/usr/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml"
#define eye_cascade "/usr/share/OpenCV/haarcascades/haarcascade_eye.xml"


void check_eye(IplImage *img) {
  int i, j;
  CvHaarClassifierCascade *cascade;
  CvSeq *eyes;
  /* 8-bit depth RGB representation of color red */
  static CvScalar RED = {0, 0, 255};
  CvMemStorage *storage;
  CvRect *r;
  IplImage *eye;
  IplImage *im_bw;
  IplImage *tmp, *tmp1;
  int offset;

  /* Load the classifier data from the .xml file */
  cascade = (CvHaarClassifierCascade *)cvLoad(eye_cascade, NULL, NULL, NULL);

  storage = cvCreateMemStorage(0);
  cvClearMemStorage(storage);

  eyes = cvHaarDetectObjects(
      img,
      cascade,
      storage,
      1.1,
      2,
      CV_HAAR_DO_CANNY_PRUNING,
      cvSize(40, 40),
      cvSize(40, 40));

  for (i = 0; i < (eyes ? eyes->total : 0); i++) {
    r = (CvRect *)cvGetSeqElem(eyes, i);
    cvSetImageROI(img, *r);
    eye = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    cvCopy(img, eye, NULL);
    //    cvResetImageROI(img);

    IplImage *im_gray = cvCreateImage(cvGetSize(eye), IPL_DEPTH_8U, 1);
    cvCvtColor(eye, im_gray, CV_RGB2GRAY);
    //    cvShowImage("gray", eye);

    im_bw = cvCreateImage(cvGetSize(im_gray), IPL_DEPTH_8U, 1);
    cvThreshold(im_gray, im_bw, 15, 255, CV_THRESH_BINARY);
    //    cvThreshold(im_gray, im_bw, 2, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    //    cvThreshold(img_src, img_dest, 128, 255, CV_THRESH_BINARY | CV_THRESH_OTSU); 

    //   cvRectangle(eye, ul, lr, RED, 3, 8, 0);

    IplImage *tmp = cvCreateImage(cvGetSize(eye), IPL_DEPTH_8U, 1);
    IplImage *tmp1 = cvCreateImage(cvGetSize(eye), IPL_DEPTH_8U, 1);
    CvSeq *contour = 0;

    cvDilate(im_bw, tmp, 0, 1);
    cvErode(tmp, tmp1, 0, 1);

    CvMemStorage* storage1 = cvCreateMemStorage(0);
    cvFindContours(tmp1, storage1, &contour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0));
    for ( ; contour != 0; contour = contour->h_next) {
      cvDrawContours(img, contour, RED, RED, -1, -CV_FILLED, 8, cvPoint(0,0));
    }

#if 0
    cvSmooth(tmp1, im_bw, CV_GAUSSIAN, 9, 9, 3, 0);
    CvSeq* circles = cvHoughCircles(tmp1, storage1, CV_HOUGH_GRADIENT, 4, tmp1->width/4, 100, 100, 1, 50);
    int i;
    for (i = 0; i < circles->total; i++) {
      float *p = (float *)cvGetSeqElem(circles, i);
      //      cvCircle(im_bw, cvPoint(cvRound(p[0]),cvRound(p[1])), 3, CV_RGB(0,255,0), -1, 8, 0 );
      //cvCircle(im_bw, cvPoint(cvRound(p[0]), cvRound(p[1])), cvRound(p[2]), CV_RGB(255, 0, 0), 3, 8, 0 );
      cvCircle(tmp1, cvPoint(cvRound(p[0]), cvRound(p[1])), cvRound(p[2]), CV_RGB(255, 0, 0), 3, 8, 0 );
    }
#endif

    cvReleaseImage(&im_gray);
    cvReleaseImage(&im_bw);
  }
  cvResetImageROI(img);
  cvReleaseImage(&eye);
  /*
  cvShowImage("img", img);

  cvWaitKey(0);

  cvDestroyWindow("img");
  */

}

int main(int argc, char *argv[])
{
  IplImage *image;
  CvHaarClassifierCascade *cascade;
  CvMemStorage *storage;
  CvSeq *faces;
  int i, j;
  CvRect *r;
  IplImage *faceImage;
  IplImage *eyesImage;
  CvPoint pt1;
  CvPoint pt2;



  /* Load the classifier data from the .xml file */
  cascade = (CvHaarClassifierCascade *)cvLoad(cascade_name, NULL, NULL, NULL);

  /* create a window with handle result */
  cvNamedWindow("result", 1);

  /* read the input image */
  image = cvLoadImage(argv[1], 1);

  /* create memory storage to be used by cvHaarDetectObjects */
  storage = cvCreateMemStorage(0);
  cvClearMemStorage(storage);
  faces = cvHaarDetectObjects(
      image,
      cascade,
      storage,
      1.1,
      2,
      CV_HAAR_DO_CANNY_PRUNING,
      cvSize(40, 40),
      cvSize(40, 40));

  /* go through all the detected faces, and draw them into the input image */
  for (i = 0; i < (faces ? faces->total : 0); i++) {
    r = (CvRect *)cvGetSeqElem(faces, i);

    pt1.x = r->x;
    pt2.x = r->x + r->width;
    pt1.y = r->y;
    pt2.y = r->y+r->height;


    cvSetImageROI(image, *r);
    faceImage = cvCreateImage(cvGetSize(image), image->depth, image->nChannels);
    cvCopy(image, faceImage, NULL);
    cvResetImageROI(image);

    eyesImage = cvCreateImage(cvGetSize(image), image->depth, image->nChannels);
    cvCopy(image, eyesImage, NULL);

    check_eye(eyesImage);
  }

  /* show the result and wait for a keystroke form user before finishing */
  // Draw the rectangle in the input image
  cvRectangle(eyesImage, pt1, pt2, CV_RGB(0,255,0), 3, 8, 0 );

  cvShowImage("eye", eyesImage);


  cvWaitKey(0);
  cvReleaseImage(&image);
  cvReleaseImage(&faceImage);
  cvReleaseImage(&eyesImage);

  cvDestroyWindow("normal");
  cvDestroyWindow("face");
  cvDestroyWindow("eye");
  return 0;
}
