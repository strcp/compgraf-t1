/* Para compilar:
 * gcc trab1.c -o trab1 `pkg-config opencv --cflags --libs` -Werror -Wall -Wno-missing-braces
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cv.h>
#include <highgui.h>

#define FACE_CASCADE "haar/haarcascade_frontalface_alt2.xml"
#define EYE_CASCADE "haar/haarcascade_eye.xml"


CvScalar RED = {0, 0, 255};
CvScalar GREEN = {0, 255, 0};
CvScalar BLUE = {255, 0, 0};

/* Esta funcao recebe uma imagem para encontrar olhos nela */

void check_eye(IplImage *img) {
  int i;
  CvHaarClassifierCascade *cascade;
  CvSeq *eyes;
  CvMemStorage *storage;
  CvRect *r;
  IplImage *eye;
  IplImage *im_bw;
  CvPoint *pt1;
  CvPoint *pt2;

  /* Carrega estrutura treinada para deteccao de olho */
  cascade = (CvHaarClassifierCascade *)cvLoad(EYE_CASCADE, NULL, NULL, NULL);

  pt1 = NULL;
  pt2 = NULL;

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

  pt1 = malloc(sizeof(CvPoint) * eyes->total);
  pt2 = malloc(sizeof(CvPoint) * eyes->total);

  /* Itera sobre a lista de olhos encontrados pelo detector usando Haar cascade */
  for (i = 0; i < (eyes ? eyes->total : 0); i++) {
    r = (CvRect *)cvGetSeqElem(eyes, i);

    /* Salva dois vértices do retangulo aonde o olho foi encontrado.
     * Essa informaçao é utilizada mais tarde para desenhar um retangulo em
     * volta de cada olho. */
    pt1[i].x = r->x;
    pt2[i].x = r->x + r->width;
    pt1[i].y = r->y;
    pt2[i].y = r->y+r->height;

    /* Utilizando ROI para operarmos apenas em cima da imagem do olho encontrada */
    cvSetImageROI(img, *r);
    eye = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    cvCopy(img, eye, NULL);

    /* FIXME: Precisamos de uma imagem cinza para binarizar? */
    IplImage *im_gray = cvCreateImage(cvGetSize(eye), IPL_DEPTH_8U, 1);
    cvCvtColor(eye, im_gray,
      CV_RGB2GRAY //Color space conversion code
    );

    /* Binarizando a imagem */
    im_bw = cvCreateImage(cvGetSize(im_gray), IPL_DEPTH_8U, 1);
    cvThreshold(
        im_gray,          //src
        im_bw,            //dst
        15,               //threshold (valores maiores disso viram max_value)
        255,              //max value
        CV_THRESH_BINARY  //thresholding type
        );

    /* Remoçao de ruídos Dilate + Erode */
    IplImage *tmp = cvCreateImage(cvGetSize(eye), IPL_DEPTH_8U, 1);
    IplImage *tmp1 = cvCreateImage(cvGetSize(eye), IPL_DEPTH_8U, 1);
    /* Maximiza a area clara da imagem para remover ruidos */
    cvDilate(im_bw, tmp, 0, 1);
    /* Maximiza a area escura da imagem */
    cvErode(tmp, tmp1, 0, 1);

    /* Invertendo a imagem para encontrar os contornos corretamente */
    cvNot(tmp1, tmp);

    /* Encontrando contornos na imagem binarizada sem ruídos */
    CvMemStorage* storage1 = cvCreateMemStorage(0);
    CvSeq *contour = 0;
    //TODO: Entender direito esses parametros
    cvFindContours(tmp, storage1, &contour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0));

    /* Pinta e vermelho cada contorno encontrado */
    for (; contour != 0; contour = contour->h_next) {
      cvDrawContours(img, contour, RED, RED, -1, CV_FILLED, 8, cvPoint(0,0));
    }

    /* Cleanup */
    cvReleaseImage(&im_gray);
    cvReleaseImage(&im_bw);
    cvReleaseImage(&tmp);
    cvReleaseImage(&tmp1);
    cvReleaseImage(&eye);
    cvResetImageROI(img);
  }

  /* Desenhando retangulo em volta dos olhos encontrados com Haar cascade */
  for (i = 0; i < (eyes ? eyes->total : 0); i++) {
    cvRectangle(img, pt1[i], pt2[i], BLUE, 2, 8, 0 );
  }

  if (pt1 != NULL)
    free(pt1);
  if (pt2 != NULL)
    free(pt2);
}

int main(int argc, char *argv[])
{
  IplImage *image;
  CvHaarClassifierCascade *cascade;
  CvMemStorage *storage;
  CvSeq *faces;
  CvRect *r;
  IplImage *faceImage;
  CvPoint pt1;
  CvPoint pt2;
  int i;

  cvNamedWindow("result", 1);
  image = cvLoadImage(argv[1], 1);

  storage = cvCreateMemStorage(0);
  cvClearMemStorage(storage);

  /* Carrega o xml com dados treinados para deteccao de face */
  cascade = (CvHaarClassifierCascade *)cvLoad(FACE_CASCADE, NULL, NULL, NULL);
  /* Utiliza Haar cascade para localizar as faces na imagem */
  faces = cvHaarDetectObjects(
      image,
      cascade,
      storage,
      1.1,  //scaleFactor
      2,    //minNeighbors
      CV_HAAR_DO_CANNY_PRUNING, //flags
      cvSize(40, 40),   //minSize
      cvSize(40, 40));  //maxSize

  /* Itera sobre a lista de faces encontradas pelo detector usando Haar cascade */
  for (i = 0; i < (faces ? faces->total : 0); i++) {
    r = (CvRect *)cvGetSeqElem(faces, i);

    /* Salva dois vértices do retangulo aonde a face foi encontrada.
     * Essa informaçao é utilizada mais tarde para desenhar um retangulo em
     * volta da face. */
    pt1.x = r->x;
    pt2.x = r->x + r->width;
    pt1.y = r->y;
    pt2.y = r->y+r->height;

    /* Utilizando ROI para fazer crop da imagem da face */
    cvSetImageROI(image, *r);
    faceImage = cvCreateImage(cvGetSize(image), image->depth, image->nChannels);
    cvCopy(image, faceImage, NULL);
    cvResetImageROI(image);

    /* Procurar e marcar os olhos encontrados */
    check_eye(image);
  }

  /* Desenha um retangulo verde em volta da face detectada */
  cvRectangle(image, pt1, pt2, GREEN,
    2, //Espessura das arestas. Valor negativo como o CV_FILLED fazem com que a funcao preencha o retangulo com a cor especificada.
    8, /* Tipo de linha
          8 - 8-connected line
          4 - 4-connected line
          CV_AA - antialiased line */
    0 // Numero de bits fracionarios nas coordenadas do ponto
    );

  cvShowImage("final", image);
  cvShowImage("face", faceImage);

  cvWaitKey(0);

  /* Cleanup */
  cvReleaseImage(&image);
  cvReleaseImage(&faceImage);

  cvDestroyWindow("final");
  cvDestroyWindow("face");

  return 0;
}
