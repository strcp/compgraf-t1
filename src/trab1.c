/* Para compilar:
 * gcc trab1.c -o trab1 `pkg-config opencv --cflags --libs`
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cv.h>
#include <highgui.h>

#define cascade_name "/usr/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml"
#define eye_cascade "/usr/share/OpenCV/haarcascades/haarcascade_eye_tree_eyeglasses.xml"



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

    cvRectangle(img, ul, lr, RED, 3, 8, 0);
  }

   for (i = r->y; i < (r->y + r->height); i++) {
     for (j = r->x; j < (r->x + r->width); j++) {
     /*
       ((uchar*)(img->imageData + i * img->widthStep))[j*3] = 0;
       ((uchar*)(img->imageData + i * img->widthStep))[j*3+1] = 0;
       ((uchar*)(img->imageData + i * img->widthStep))[j*3+2] = 0;
    */
     }
   }
}

int main(int argc, char *argv[])
{
  IplImage *image;
  CvHaarClassifierCascade *cascade;
//  IplImage *gray;
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
