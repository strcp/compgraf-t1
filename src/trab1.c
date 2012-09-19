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
//#define eye_cascade "/usr/share/OpenCV/haarcascades/haarcascade_eye_tree_eyeglasses.xml"



void check_eye(IplImage *img) {
  int i, j;
  CvHaarClassifierCascade *cascade;
  CvSeq *eyes;
  /* 8-bit depth RGB representation of color red */
  static CvScalar RED = {0, 0, 255};
  CvPoint ul;
  CvPoint lr;
  CvMemStorage *storage;
  CvRect *r;
  IplImage *eye;
  IplImage *im_bw;
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

  /* used cvHaarDetectObjects */

  for (i = 0; i < (eyes ? eyes->total : 0); i++) {
    r = (CvRect *)cvGetSeqElem(eyes, i);
    ul.x = r->x;
    ul.y = r->y;
    lr.x = r->x + r->width;
    lr.y = r->y + r->height;

    cvSetImageROI(img, *r);
    eye = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    cvCopy(img, eye, NULL);
    cvResetImageROI(img);

    IplImage *im_gray = cvCreateImage(cvGetSize(eye), IPL_DEPTH_8U, 1);
    cvCvtColor(eye, im_gray, CV_RGB2GRAY);
    cvShowImage("gray", eye);

    im_bw = cvCreateImage(cvGetSize(im_gray), IPL_DEPTH_8U, 1);
    cvThreshold(im_gray, im_bw, 15, 255, CV_THRESH_BINARY);

#if 0
    for (i = 0; i < im_bw->height; i++) {
      for (j = 0; j < im_bw->width; j++) {
        offset = (i * im_bw->width);
        printf("%d ", ((uchar *)(im_bw->imageData))[j + offset]);
//        ((uchar *)(im_bw->imageData))[j + offset] =
      }
    }
    printf("\n");
#endif

    CvMemStorage* storage1 = cvCreateMemStorage(0);
//    cvSmooth(im_bw, im_bw, CV_GAUSSIAN, 9, 9, 3, 0);
    CvSeq* circles = cvHoughCircles(im_bw, storage1, CV_HOUGH_GRADIENT, 4, im_bw->width/4, 100, 100, 0, 50);
    int i;
    for( i = 0; i < circles->total; i++ )
    {
      float* p = (float*)cvGetSeqElem( circles, i );
      cvCircle(im_bw, cvPoint(cvRound(p[0]),cvRound(p[1])), 3, CV_RGB(0,255,0), -1, 8, 0 );
      cvCircle(im_bw, cvPoint(cvRound(p[0]),cvRound(p[1])), cvRound(p[2]), CV_RGB(255,0,0), 3, 8, 0 );
    }

    cvShowImage("bw", im_bw);

    cvWaitKey(0);
    cvReleaseImage(&im_gray);
    cvReleaseImage(&im_bw);
    cvReleaseImage(&eye);

    cvDestroyWindow("gray");
    cvDestroyWindow("bw");

//    cvRectangle(img, ul, lr, RED, 3, 8, 0);
  }

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

    cvSetImageROI(image, *r);
    faceImage = cvCreateImage(cvGetSize(image), image->depth, image->nChannels);
    cvCopy(image, faceImage, NULL);
    cvResetImageROI(image);

    eyesImage = cvCreateImage(cvGetSize(image), image->depth, image->nChannels);
    cvCopy(image, eyesImage, NULL);

    check_eye(eyesImage);
  }

  /* show the result and wait for a keystroke form user before finishing */
  cvShowImage("normal", image);
  cvShowImage("face", faceImage);
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
