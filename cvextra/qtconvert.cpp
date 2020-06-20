//#include "qtconvert.hpp"

//using namespace std;
//using namespace cv;

//Mat cvx::qt::toMat(QImage const& image, bool cloneImageData)
//{
//    switch (image.format())
//    {
//    case QImage::Format_RGB32:
//    {
//        Mat mat(image.height(),
//                image.width(),
//                CV_8UC4,
//                const_cast<uchar*>(image.bits()), image.bytesPerLine());

//        return (cloneImageData ? mat.clone() : mat);
//    }
//    case QImage::Format_RGB888:
//    {
//        QImage swapped = image.rgbSwapped();
//        return Mat(swapped.height(),
//                   swapped.width(),
//                   CV_8UC3,
//                   const_cast<uchar*>(swapped.bits()), swapped.bytesPerLine()).clone();
//    }
//    case QImage::Format_Indexed8:
//    {
//        Mat mat(image.height(),
//                image.width(),
//                CV_8UC1,
//                const_cast<uchar*>(image.bits()), image.bytesPerLine());
//        return (cloneImageData ? mat.clone() : mat);
//    }

//    default:
//        break;
//    }

//    return Mat();
//}

//Mat cvx::qt::toMat(QPixmap const& pixmap, bool cloneImageData)
//{
//    return cvx::qt::toMat(pixmap.toImage(), cloneImageData);
//}

//QColor cvx::qt::toQColor(Scalar colorBGR)
//{
//    return QColor(colorBGR[2], colorBGR[1], colorBGR[0]);
//}

//QImage cvx::qt::toQImage(InputArray src)
//{
//    Mat mat = src.getMat();

//    switch (mat.type())
//    {

//    case CV_8UC4:
//    {
//        return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB32);
//    }
//    case CV_8UC3:
//    {
//        return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888).rgbSwapped();
//    }
//    case CV_8UC1:
//    {
//        static QVector<QRgb> sColorTable;
//        if ( sColorTable.isEmpty() )
//        {
//            for (int i = 0; i < 256; ++i)
//            {
//                sColorTable.push_back(qRgb(i, i, i));
//            }
//        }

//        QImage image(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Indexed8);
//        image.setColorTable(sColorTable);
//        return image;
//    }

//    default:
//        break;
//    }

//    return QImage();
//}

//QPixmap cvx::qt::toQPixmap(InputArray src)
//{
//    return QPixmap::fromImage(toQImage(src));
//}
