/* Para compilar:
 * gcc teste.c -o teste -I/usr/include/opencv -lopencv_core -lopencv_imgproc
 * -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d
 * -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy
 * -lopencv_flann */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cv.h>
#include <highgui.h>

//#define cascade_name "/usr/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml"
#define cascade_name "/usr/share/OpenCV/haarcascades/haarcascade_eye_tree_eyeglasses.xml"


IplImage *image;

void check_eye(CvRect *r) {
  int i, j;

   for (i = r->y; i < (r->y + r->height); i++) {
     for (j = r->x; j < (r->x + r->width); j++) {
       ((uchar*)(image->imageData + i * image->widthStep))[j*3] = 0;
       ((uchar*)(image->imageData + i * image->widthStep))[j*3+1] = 0;
       ((uchar*)(image->imageData + i * image->widthStep))[j*3+2] = 0;
     }
   }
}

int main(int argc, char *argv[])
{
  CvHaarClassifierCascade *cascade;
  IplImage *img2;
//  IplImage *gray;
  CvMemStorage *storage;
  CvSeq *faces;
  int i, j;
  CvRect *r;

  /* Load the classifier data from the .xml file */
  cascade = (CvHaarClassifierCascade *)cvLoad(cascade_name, NULL, NULL, NULL);

  /* create a window with handle result */
  cvNamedWindow("result", 1);

  /* read the input image */
  image = cvLoadImage(argv[1], 1);

#if 0
  /*
     create a b&w image of the same size object as a placeholder for now
     - cvSize(width, height) is the object representing dimensions of the image
     - 8 stands for 8-bit depth
     - 1 means one-channel (b&w)
     Hence a 24bit RGB image of size 800x600 would be created with cvCreateImage( cvSize(800, 600), 8, 3);
   */
  gray = cvCreateImage(cvSize(image->width, image->height), 8, 1);

  /* now convert the input image into b&w and store it in the placeholder */
  cvCvtColor(image, gray, CV_BGR2GRAY);
#endif
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

  /* used cvHaarDetectObjects */

  /* 8-bit depth RGB representation of color red */
  static CvScalar RED = {0, 0, 255};

  /* go through all the detected faces, and draw them into the input image */
  for (i = 0; i < (faces ? faces->total : 0); i++) {
    r = (CvRect *)cvGetSeqElem(faces, i);
    CvPoint ul;
    CvPoint lr;
    ul.x = r->x;
    ul.y = r->y;
    lr.x = r->x + r->width;
    lr.y = r->y + r->height;

    check_eye(r);

   //cvSetImageROI(image, *r);

//   cvCopy(image, img2, NULL);
   //cvResetImageROI(image);

    //IplImage *img2 = cvCreateImage(cvGetSize(image), image->depth, image->nChannels);
//    break;

    /* draws a rectangle with given coordinates of the upper left
       and lower right corners into an image */
    //cvRectangle(image, ul, lr, RED, 3, 8, 0);
  }
  /*
  for (i = r->y; i < (r->y + r->height); i++) {
    for (j = r->x; j < (r->x + r->width); j++) {
      ((uchar*)(image->imageData + i * image->widthStep))[j*3] = 0;
      ((uchar*)(image->imageData + i * image->widthStep))[j*3+1] = 0;
      ((uchar*)(image->imageData + i * image->widthStep))[j*3+2] = 0;
    }
  }
  */
#if 0
  /* free up the memory */
  cvReleaseImage(&gray);
#endif

  /* show the result and wait for a keystroke form user before finishing */
  cvShowImage("result", image);

  cvWaitKey(0);
  cvReleaseImage(&image);
  cvDestroyWindow("result");

  return 0;
}
