/* Para compilar:
 * gcc trab1.c -o trab1 `pkg-config opencv --cflags --libs`
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cv.h>
#include <highgui.h>

#define FACE_CASCADE "/usr/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml"
#define EYE_CASCADE "/usr/share/OpenCV/haarcascades/haarcascade_eye.xml"


static CvScalar RED = {0, 0, 255};
static CvScalar GREEN = {0, 255, 0};

void check_eye(IplImage *img) {
  int i, j;
  CvHaarClassifierCascade *cascade;
  CvSeq *eyes;
  CvMemStorage *storage;
  CvRect *r;
  IplImage *eye;
  IplImage *im_bw;
  IplImage *tmp, *tmp1;
  int offset;

  cascade = (CvHaarClassifierCascade *)cvLoad(EYE_CASCADE, NULL, NULL, NULL);

  storage = cvCreateMemStorage(0);
  cvClearMemStorage(storage);

  /* Utiliza Haar cascade para localizar os olhos na imagem */
  eyes = cvHaarDetectObjects(
      img,
      cascade,
      storage,
      1.1,
      2,
      CV_HAAR_DO_CANNY_PRUNING,
      cvSize(40, 40),
      cvSize(40, 40));

  /* Itera sobre a lista de olhos encontrados pelo detector usando Haar cascade */
  for (i = 0; i < (eyes ? eyes->total : 0); i++) {
    r = (CvRect *)cvGetSeqElem(eyes, i);
    cvSetImageROI(img, *r);
    eye = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    cvCopy(img, eye, NULL);

    /* FIXME: Precisamos de uma imagem cinza para binarizar? */
    IplImage *im_gray = cvCreateImage(cvGetSize(eye), IPL_DEPTH_8U, 1);
    cvCvtColor(eye, im_gray, CV_RGB2GRAY);

    /* Binarizando a imagem */
    im_bw = cvCreateImage(cvGetSize(im_gray), IPL_DEPTH_8U, 1);
    cvThreshold(im_gray, im_bw, 15, 255, CV_THRESH_BINARY);

    /* Remoçao de ruídos Dilate + Erode */
    IplImage *tmp = cvCreateImage(cvGetSize(eye), IPL_DEPTH_8U, 1);
    IplImage *tmp1 = cvCreateImage(cvGetSize(eye), IPL_DEPTH_8U, 1);
    cvDilate(im_bw, tmp, 0, 1);
    cvErode(tmp, tmp1, 0, 1);

    /* Encontrando contornos na imagem binarizada sem ruídos */
    CvMemStorage* storage1 = cvCreateMemStorage(0);
    CvSeq *contour = 0;
    cvFindContours(tmp1, storage1, &contour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0));
    for (; contour != 0; contour = contour->h_next) {
      cvDrawContours(img, contour, RED, RED, -1, -CV_FILLED, 8, cvPoint(0,0));
    }

    /* Cleanup */
    cvReleaseImage(&im_gray);
    cvReleaseImage(&im_bw);
    cvReleaseImage(&tmp);
    cvReleaseImage(&tmp1);
  }

  cvResetImageROI(img);
  cvReleaseImage(&eye);
}

int main(int argc, char *argv[])
{
  IplImage *image;
  CvHaarClassifierCascade *cascade;
  CvMemStorage *storage;
  CvSeq *faces;
  CvRect *r;
  IplImage *faceImage;
  IplImage *eyesImage;
  CvPoint pt1;
  CvPoint pt2;
  int i;

  cvNamedWindow("result", 1);
  image = cvLoadImage(argv[1], 1);

  storage = cvCreateMemStorage(0);
  cvClearMemStorage(storage);

  cascade = (CvHaarClassifierCascade *)cvLoad(FACE_CASCADE, NULL, NULL, NULL);
  /* Utiliza Haar cascade para localizar as faces na imagem */
  faces = cvHaarDetectObjects(
      image,
      cascade,
      storage,
      1.1,
      2,
      CV_HAAR_DO_CANNY_PRUNING,
      cvSize(40, 40),
      cvSize(40, 40));

  /* Itera sobre a lista de faces encontradas pelo detector usando Haar cascade */
  for (i = 0; i < (faces ? faces->total : 0); i++) {
    r = (CvRect *)cvGetSeqElem(faces, i);

    /* Cria dois vértices do retangulo aonde a face foi encontrada.
     * Essa informaçao é utilizada mais tarde para desenhar um retangulo em
     * volta da face. */
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

  /* Desenha um retangulo verde em volta da face detectada */
  cvRectangle(eyesImage, pt1, pt2, GREEN, 2, 8, 0 );

  cvShowImage("eye", eyesImage);

  cvWaitKey(0);

  /* Cleanup */
  cvReleaseImage(&image);
  cvReleaseImage(&faceImage);
  cvReleaseImage(&eyesImage);

  cvDestroyWindow("normal");
  cvDestroyWindow("face");
  cvDestroyWindow("eye");

  return 0;
}
