//________OPENCV__________________________//
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv/cv.h>


//_____STL__________//
#include <fstream>
#include <cmath>
#include <sys/types.h>
#include <dirent.h>


//_____OpenMP_______//
#include <omp.h>





//______________________________________________Variables globales_________________________________________________________

/*rc representa el radio que debe tener el circulo resultante de aplicar la transformación afín a cada elipse tal que cubra un cuadrado de lados szRectxszRect*/
int szRect=40;
int ARect=szRect*szRect;
double rc=sqrt(2*(ARect))/2;
double centerRect=szRect/2;


int numberOrientations=4;//Número de orientaciones
cv::Mat *gaborKernel=NULL;
int *pow3=NULL;
double ut=0;

const int maxFeatures=1;
int dHist=4*4*81;
cv::Mat *imgHist=new cv::Mat(maxFeatures,dHist,CV_32F);


//__________________________________________________________________________________________________________________________



void zScoreNormalization(const cv::Mat * const imageSrc,cv::Mat *imageDest)
{



cv::Mat u,s;
cv::Mat srcTemp;
imageSrc->assignTo(srcTemp,CV_64F);

cv::meanStdDev(srcTemp,u,s);
(*imageDest)=(srcTemp-(u.at<double>(0,0))+3*(s.at<double>(0,0)))/(6*s.at<double>(0,0));


double *p_imageDest=imageDest->ptr<double>(0);

for(int j=0;j<ARect;j++){
if(p_imageDest[j]<0)p_imageDest[j]=0;
else if(p_imageDest[j]>1)p_imageDest[j]=1;

}

}





void constructGaborKernels()
{

/*NOTA:Se trabajo con precisión de 32 bits debido a que la función filter2D de openCV requiere un kernel de tipo CV_32F,
se aclara también que no se optimizo este método debido a que solo se llama una vez en el constructor y los cálculos requeridos 
son insignificantes*/

double orientations[]={0,45,90,135};//Vector de orientaciones en grados del filtro
double sigma=1;
double Kv=M_PI/2;
int szGk=6*sigma+1;//Se sumo 1 para que el kernel sea impar 

/*Creamos el dominio de evaluación*/
/*Esta parte del código imita la función meshgrid de MATLAB*/
cv::Mat X(szGk,szGk,cv::DataType<int>::type);
cv::Mat Y(szGk,szGk,cv::DataType<int>::type);

int lim=-szGk/2;
for(int i=0;i<szGk;i++,lim++)
X.col(i)=lim;
Y=X.clone().t();
/*________________________________*/

cv::Mat Z=(X.mul(X)+Y.mul(Y));
Z.convertTo(Z,CV_32F);

//__________CALCULANDO LA GAUSIANA__________//
cv::Mat GAUSIAN;
cv::exp(-((Kv*Kv)/(2*sigma*sigma))*Z,GAUSIAN);
GAUSIAN=((Kv*Kv)/(sigma*sigma))*GAUSIAN;

std::cout<<"Tipo gausiana="<<GAUSIAN.type()<<"\n";

//_______Aquí se eliminan valores producto del ruido numérico_____________________
float max_GAUSIAN=*std::max_element(GAUSIAN.begin<float>(),GAUSIAN.end<float>());
float EPS=std::numeric_limits<float>::epsilon();
float aux2=EPS*max_GAUSIAN;

for(int i=0;i<GAUSIAN.rows;i++){
for(int j=0;j<GAUSIAN.cols;j++){
if(GAUSIAN.at<float>(i,j)<aux2){
GAUSIAN.at<float>(i,j)=0;
}
}
}
//_________________________________________________________________________________

//___________Aquí se normaliza la gausiana para que su suma sea 1__________________
float sum_GAUSIAN=cv::sum(GAUSIAN)[0];
if(sum_GAUSIAN!=0)
GAUSIAN=GAUSIAN/sum_GAUSIAN;
//_________________________________________________________________________________

//__________________________________________//

gaborKernel=new cv::Mat[numberOrientations];//vector el kernel para cada diferente orientación
//Se transforman a flotantes debido a que las operaciones que vienen son con matrices tipo flotante
X.convertTo(X,CV_32F);
Y.convertTo(Y,CV_32F);

for(int n=0;n<numberOrientations;n++)
{

double alpha=(orientations[n])*M_PI/180;
cv::Mat Xp=Kv*cos(alpha)*X.clone();
cv::Mat Yp=Kv*sin(alpha)*Y.clone();

//_________PRIMERO CONSTRUIMOS LA FUNCION SIN 2D_________//
cv::Mat sin2D(szGk,szGk,CV_32F);
for(int i=0;i<szGk;i++){
for(int j=0;j<szGk;j++){
sin2D.at<float>(i,j)=sin(Xp.at<float>(i,j)+Yp.at<float>(i,j));
}
}

//________________________________________________________//

//Ahora almacenamos cada filtro
gaborKernel[n]=GAUSIAN.mul(sin2D);
}


//Aquí se deja inicializado el vector pow3 para ser usado en la función gtp(cv::Mat *img)
pow3=new int[numberOrientations];
for(int i=0;i<numberOrientations;i++)
 pow3[i]=std::pow(3,i);

//Aquí dejamos inicializado el umbral usado en el patrón ternario (gtp(cv::Mat *img))
ut=0.007;

}



void gtp(cv::Mat *img,cv::Mat *ltp)
{

cv::Mat temp;
(*ltp)=cv::Mat::zeros(szRect,szRect,CV_8UC1);

for(int i=0;i<numberOrientations;i++)
{
/*NOTA:Matlab y openCV pueden presentar leves diferencias en los bordes de la imagen resultado temp, recuerde también que el resultado de
el filtrado es una correlación y por lo tanto da el negativo de la convolución (presentándose el dilema ¿escojo el +, osea el negativo del resultado de filter2D que seria igual a la convolución o el positivo, es decir simplemente filter2D?), puesto que lo que se va a construir es una codificación ternaria el signo no importa ya que simplemente la codificación se reorganiza*/

cv::filter2D(*img,temp,CV_64F,gaborKernel[i]);
(*ltp)=(*ltp)+(pow3[i]*((temp<-ut)+2*(temp>ut))/255);

}

}

//___________________________________________________________________________________________________
namespace Gh{//Este namespace se crea para facilitar el uso de la función calcHist

  //Número de bins
  int histSize[]={81};

  //Rango de valores del histograma
  float range[] = {0,81};//Recuerde que el limite superior es exclusivo
  const float* histRange[] = { range };

  //Configuraciones (ver documentación openCV)
  bool uniform = true; 
  bool accumulate =false;

  //Canales a analizar
  int channels[] = {0};
}


void histogram(cv::Mat *img)
{

static int row=0;
/*Esta parte es específicamente para imágenes img de tamaño 40x40*/
cv::Mat temp,hist;

int nh=0;
for(int i=0;i<4;i++){
 for(int j=0;j<4;j++){
 
 temp=((*img)(cv::Range(10*i,10*i+10),cv::Range(10*j,10*j+10))).clone(); 
 //A continuación extraemos el histograma de la imagen temp
 calcHist(&temp,1,0,cv::Mat(),hist,1,Gh::histSize,Gh::histRange,Gh::uniform,Gh::accumulate);
 //A continuación concatenamos el histograma con los previamente almacenados 
 imgHist->row(row).colRange(nh,nh+81)=hist.t();
 nh=nh+81; 

 }
}


imgHist->row(row)=imgHist->row(row)/100;//Aquí se normaliza el histograma (recuerde cada celda mide 10x10=100)

float *prow=imgHist->row(row).ptr<float>(0);
for(int i=0;i<1296;i++)
prow[i]=tanh(prow[i]);

}



bool mountRamdisk()
{

int VAL=0;

 DIR* dir = opendir("/ramdisk_UVface");
  if (dir)
  {
   
    closedir(dir);

    VAL=system("cp ./ramdisk_inicio/extract_features_64bit.ln /ramdisk_UVface/extract_features_64bit.ln");
    VAL=system("cp ./ramdisk_inicio/imagen.pgm.sedgelap /ramdisk_UVface/imagen.pgm.sedgelap");
    VAL=system("cp ./ramdisk_inicio/imagen.pgm /ramdisk_UVface/imagen.pgm");

   std::cout<<"Los archivos de inicio necesarios para el funcionamiento adecuado del descriptor GTP se montaron adecuadamente en la carpeta /ramdisk_UVface\n";
   return true;
  } else
  {
    std::cout<<"La carpeta ramdisk_UVface necesaria para montar algunos archivos de inicio, necesarios para la ejecución adecuada del descriptor GTP, no existe o no se tienen los permisos necesarios para escribir sobre ella, por tal razón el programa no puede ejecutarse.\n";
    
    return false;
  }


}






int main( int, char** argv )
{

 
if(!mountRamdisk()) return 0; //SI la carpeta ramdisk_UVface no existe debe pararse el flujo del programa


cv::namedWindow("Imagen", CV_WINDOW_FREERATIO);
cv::namedWindow("Region Afin", CV_WINDOW_NORMAL);
cv::namedWindow("Region Afin Transformada", CV_WINDOW_NORMAL);
cv::namedWindow("LTP", CV_WINDOW_NORMAL);
cv::namedWindow("HISTOGRAMA", CV_WINDOW_FREERATIO);


constructGaborKernels(); //INicializando el kernel de GABOR
//Para mostrar el dibujo del rectángulo, rectángulo rotado y la elipse 
bool showRect=false;            
bool showCenterPoint=true;
bool showRectRotate=false;
//Para mostrar todas las elipses o solo las que están dentro de la imagen.
bool showEllipces=false;
bool showEllipcesOutImage=false;
//Mantener objeto graficado
bool showAll=true;

cv::Mat imgGray, imgShow;
double vec_dp[10000][5];
cv::Mat MQ(2,2,CV_64F);
double* MQP = MQ.ptr<double>(0);
cv::Mat EIG_val(2,1,CV_64F);
cv::Mat EIG_vec(2,2,CV_64F);

cv::Rect brect;
cv::RotatedRect rRect;
std::vector<cv::Rect> rectang;
cv::Mat *imgRect=new cv::Mat;
cv::Mat *imgRectShow=new cv::Mat;
cv::Mat *imgRectZscore=new cv::Mat;
cv::Mat *imgRectGtp=new cv::Mat(szRect,szRect,CV_8UC1);
cv::Mat *LTP=new cv::Mat;
cv::Mat e(2,2,CV_64F);
e.at<double>(1,0)=0;
e.at<double>(0,1)=0;
cv::Mat pc(2,1,CV_64F);
cv::Mat pcn(2,1,CV_64F);
cv::Mat iM2(2,3,CV_64F);
cv::Mat iM1(2,2,CV_64F);
cv::Mat Rimg, RimgShow;

int histSize =1296;
int hist_w =2*1296; int hist_h =16*81;
int bin_w =cvRound( (double) hist_w/histSize);
cv::Mat histImage( hist_h, hist_w, CV_8UC3,cv::Scalar( 0,0,0) );



std::ifstream fp;
fp.open("/ramdisk_UVface/imagen.pgm.sedgelap",std::ifstream::in);

imgShow=cv::imread("/ramdisk_UVface/imagen.pgm");
cv::cvtColor(imgShow,imgGray, CV_BGR2GRAY);


//______________________________CanAff________________________________________________
FILE* pipe = popen("/ramdisk_UVface/extract_features_64bit.ln -sedgelap -noangle -i /ramdisk_UVface/imagen.pgm", "r");




    char buffer[128];
    std::string result = "";
    int cl=0;
    while(!feof(pipe)) {
    	if(fgets(buffer, 128, pipe) != NULL){
    		//result += buffer;
                if(cl==4)
                result+=buffer+16;
                //printf("%s\n",buffer+16);
                cl++;  
                }
    }

fclose(pipe);

//____________________________________________________________________________________


int numEllipcesInImage=0;

if(atoi(result.c_str())>0){

fp.seekg(std::ios_base::beg);
double dp;
int num;
fp>>dp;
fp>>num;
//std::cout<<dp<<"\n";
//std::cout<<num<<"\n";

std::cout<<"Número de puntos clave: "<<num<<"\n";

for(int i=0;i<num;i++)
{
for(int j=0;j<5;j++)
fp>>(vec_dp[i][j]);
}



cv::Mat temp=imgShow.clone();


for(int i=0;i<num;i++)
{

if(!showAll)
imgShow=temp.clone();


MQP[0]=vec_dp[i][2];
MQP[1]=vec_dp[i][3];
MQP[2]=vec_dp[i][3];
MQP[3]=vec_dp[i][4];

eigen(MQ,EIG_val,EIG_vec);
double l1=1.0/sqrt(EIG_val.at<double>(1,0));//Eje mayor, puesto que se divide entre el eigenvector menor
double l2=1.0/sqrt(EIG_val.at<double>(0,0));//Eje menor,puesto que se divide entre el eigenvector mayor


double alpha=atan2(EIG_vec.at<double>(1,1),EIG_vec.at<double>(1,0));//Se toma como referencia el eje mayor



double c=vec_dp[i][0];
double f=vec_dp[i][1];



//_____Coordenadas que encierran la elipse____________________//
double xxi,yyi,xxf,yyf,xsi,xsf,ysi,ysf,sc;
sc=MQP[2]/sqrt(MQP[0]*MQP[3]-MQP[2]*MQP[2]);


xsi=sc*sqrt(1/MQP[0]);
yyi=-(MQP[0]/MQP[1])*xsi;
xsi=xsi+c;
yyi=yyi+f;


xsf=-sc*sqrt(1/MQP[0]);
yyf=-(MQP[0]/MQP[1])*xsf;
xsf=xsf+c;
yyf=yyf+f;


ysi=sc*sqrt(1/MQP[3]);
xxi=-(MQP[3]/MQP[1])*ysi;
xxi=xxi+c;
ysi=ysi+f;


ysf=-sc*sqrt(1/MQP[3]);
xxf=-(MQP[3]/MQP[1])*ysf;
xxf=xxf+c;
ysf=ysf+f;

//_______________________________________________________________________________//

if(showCenterPoint)
imgShow.at<cv::Vec3b>(f,c)=cv::Vec3b(0,0,255);


rRect=cv::RotatedRect(cv::Point2f(c,f), cv::Size2f(2*l1,2*l2),alpha*(180/M_PI));


if(showRectRotate){
cv::Point2f vertices[4];
rRect.points(vertices);
for (int i = 0; i < 4; i++)
cv::line(imgShow, vertices[i], vertices[(i+1)%4],cv::Scalar(0,255,0));
}



brect=cv::Rect(cv::Point(xxi,yyi),cv::Point(xxf,yyf));

if(showRect)
cv::rectangle(imgShow,brect,cv::Scalar(0,0,255));

if((xxi>=0)&&(yyi>=0)&&(xxf<imgShow.cols)&&(yyf<imgShow.rows)){

rectang.push_back(cv::Rect(cv::Point(xxi,yyi),cv::Point(xxf,yyf)));

if(showEllipces)
ellipse(imgShow,cv::Point(c,f),cv::Size(l1,l2),alpha*(180/M_PI),0,360,cv::Scalar(255,0,0));


Rimg=imgGray(brect);
RimgShow=imgShow(brect);


e.at<double>(0,0)=std::sqrt(EIG_val.at<double>(0,0));
e.at<double>(1,1)=std::sqrt(EIG_val.at<double>(1,0));
iM1=rc*EIG_vec.t()*e*EIG_vec;



//Centro de la imagen sin transformar
pc.at<double>(0,0)=Rimg.cols/2;
pc.at<double>(1,0)=Rimg.rows/2;

/*Ubicamos la primera fila de iM1 en las correspondientes ubicaciones en iM2*/
iM2.at<double>(0,0)=iM1.at<double>(0,0);
iM2.at<double>(0,1)=iM1.at<double>(0,1);

pcn=centerRect-(iM1*pc);/*Aquí se calcula la traslación necesaria para que la imagen quede centrada, (*iM1)*(*pc) representa el nuevo centro de la imagen, y centerRect es el centro de la imagen szRectxszRect, de tal forma que centerRect-(*iM1)*(*pc) es el vector que trasladara la imagen transformada por la transformación afín iM1 de tal forma que esta quede centrada con respecto al rectángulo szRectxszRect*/
iM2.at<double>(0,2)=pcn.at<double>(0,0);
iM2.at<double>(1,0)=iM1.at<double>(1,0);
iM2.at<double>(1,1)=iM1.at<double>(1,1);
iM2.at<double>(1,2)=pcn.at<double>(1,0);

cv::warpAffine(Rimg,*imgRect,iM2,cv::Size(szRect,szRect),cv::INTER_LANCZOS4);


/*Ubicamos la primera fila de iM1 en las correspondientes ubicaciones en iM2*/
iM1=2*iM1;
iM2.at<double>(0,0)=iM1.at<double>(0,0);
iM2.at<double>(0,1)=iM1.at<double>(0,1);
pcn=100-(iM1*pc);/*Aquí se calcula la traslación necesaria para que la imagen quede centrada, (*iM1)*(*pc) representa el nuevo centro de la imagen, y centerRect es el centro de la imagen szRectxszRect, de tal forma que centerRect-(*iM1)*(*pc) es el vector que trasladara la imagen transformada por la transformación afín iM1 de tal forma que esta quede centrada con respecto al rectángulo szRectxszRect*/
iM2.at<double>(0,2)=pcn.at<double>(0,0);
iM2.at<double>(1,0)=iM1.at<double>(1,0);
iM2.at<double>(1,1)=iM1.at<double>(1,1);
iM2.at<double>(1,2)=pcn.at<double>(1,0);
cv::warpAffine(RimgShow,*imgRectShow,iM2,cv::Size(5*szRect,5*szRect),cv::INTER_LANCZOS4);
//A continuación se normaliza la iluminación de cada imagen del vector a cv::Mat imagesRect con el metodo zScoreNormalization


zScoreNormalization(imgRect,imgRectZscore); 
gtp(imgRectZscore,imgRectGtp);//Recuerde que la devolución es de tipo CV_8UC1
histogram(imgRectGtp);/*Se procede a calcular el histograma por cada imagen y almacenarlo en la fila row de imgHist*/



//______________________Mostrar LTP________________________________//
double min;
double max;
cv::minMaxIdx((*imgRectGtp), &min, &max);
cv::Mat auxTemp=(*imgRectGtp).clone();
auxTemp.convertTo(auxTemp,CV_32F);
auxTemp=255*((auxTemp)-min)/(max-min);
auxTemp.convertTo(auxTemp,CV_8UC1);
*LTP=255*((*imgRectGtp)-min)/(max-min);
//__________________________________________________________________//


//______________________Mostrar histograma__________________________//
histImage.setTo(cv::Scalar(255,255,255));
cv::normalize(*imgHist, *imgHist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
for( int ii = 1; ii < histSize; ii++ )
{
    cv::line( histImage, cv::Point( bin_w*(ii-1), hist_h - cvRound(imgHist->at<float>(ii-1)) ) ,
                     cv::Point( bin_w*(ii), hist_h - cvRound(imgHist->at<float>(ii)) ),
                     cv::Scalar( 255, 0, 0), 2, 8, 0  );

}

//__________________________________________________________________//

numEllipcesInImage++;

}else
{

if(showEllipcesOutImage)
ellipse(imgShow,cv::Point(c,f),cv::Size(l1,l2),alpha*(180/M_PI),0,360,cv::Scalar(0,0,255));

}



cv::imshow("Imagen",imgShow);

if(!RimgShow.empty()){
cv::imshow("Region Afin",RimgShow);
cv::imshow("Region Afin Transformada",*imgRectShow);
cv::imshow("LTP",*LTP);
cv::imshow("HISTOGRAMA",histImage);

}

std::cout<<"Total elipses "<<(i+1)<<"\n";

if(27==cv::waitKey(0))
{

delete imgHist;
delete []gaborKernel;
delete []pow3;
delete imgRect;
delete imgRectShow;
delete imgRectZscore;
delete imgRectGtp;
delete LTP;
fp.close();

return 0;
}



}

}

std::cout<<"TOTAL DE PUNTOS  CLAVE ÚTILES "<<numEllipcesInImage<<"\n";











delete imgHist;
delete []gaborKernel;
delete []pow3;
delete imgRect;
delete imgRectShow;
delete imgRectZscore;
delete imgRectGtp;
delete LTP;
fp.close();


return 0;
}
























