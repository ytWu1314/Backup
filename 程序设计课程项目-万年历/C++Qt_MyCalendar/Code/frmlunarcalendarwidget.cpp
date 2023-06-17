#include "frmlunarcalendarwidget.h"
#include "ui_frmlunarcalendarwidget.h"
#include<QPainter>
#include <QGraphicsOpacityEffect>
#include <QFile>
#include <QFileDialog>

frmLunarCalendarWidget::frmLunarCalendarWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::frmLunarCalendarWidget)
{
    ui->setupUi(this);
    this->initForm();
}

frmLunarCalendarWidget::~frmLunarCalendarWidget()
{
    delete ui;
}

void frmLunarCalendarWidget::initForm()
{
    //设置背景
    QPixmap bgPixmap("../image/peking_university.png");
    QLabel *bgLabel = new QLabel(this);
    bgLabel->setGeometry(0, 0, this->width(), this->height());
    bgLabel->lower();

//    //设置日历面板透明度
    QGraphicsOpacityEffect *opacityEffect=new QGraphicsOpacityEffect;
    ui->lunarCalendarWidget->setGraphicsEffect(opacityEffect);
    opacityEffect->setOpacity(0.85);

    //设置textEdit透明度
    QGraphicsOpacityEffect *opacityEffect2=new QGraphicsOpacityEffect;
    ui->textEdit->setGraphicsEffect(opacityEffect2);
    opacityEffect2->setOpacity(0.7);
    ui->cboxWeekNameFormat->setCurrentIndex(2);
}

void frmLunarCalendarWidget::on_cboxCalendarStyle_currentIndexChanged(int index)
{
    ui->lunarCalendarWidget->setCalendarStyle((LunarCalendarWidget::CalendarStyle)index);
}

void frmLunarCalendarWidget::on_cboxSelectType_currentIndexChanged(int index)
{
    ui->lunarCalendarWidget->setSelectType((LunarCalendarWidget::SelectType)index);
}

void frmLunarCalendarWidget::on_cboxWeekNameFormat_currentIndexChanged(int index)
{
    ui->lunarCalendarWidget->setWeekNameFormat((LunarCalendarWidget::WeekNameFormat)index);
}

void frmLunarCalendarWidget::on_ckShowLunar_stateChanged(int arg1)
{
    ui->lunarCalendarWidget->setShowLunar(arg1 != 0);
}

void frmLunarCalendarWidget::paintEvent(QPaintEvent *)
{
    QPainter p;
    p.begin(this);

    p.drawPixmap(0,0,width(),height(),QPixmap("../image/peking_university.png"));
    p.end();
}

void frmLunarCalendarWidget::on_pushButton_clicked()
{
    //定义路径
    QString path =QFileDialog ::getOpenFileName(this,
                                                "open","../","TXT(*.txt)");
    if(path.isEmpty()==false)
    {
        //文件对象
        QFile file(path);

        //打开文件，只读方式
        bool isok=file.open(QIODevice::ReadOnly);

        if(isok)
        {
            //读文件，默认识别utf8编码
            QByteArray array= file.readAll();

            //显示到编辑区
            ui->textEdit->setText(array);
        }
        //关闭文件
        file.close();
    }
}


void frmLunarCalendarWidget::on_pushButton_2_clicked()
{
    QString path =QFileDialog::getSaveFileName(this,"save","../","TXT(*.txt)");
    if(path.isEmpty()==false )
    {
        QFile file (path); //创建文件对像，关联文件名字

        bool isok= file.open(QIODevice::WriteOnly); //打开文件，只写方式

        if(isok)
        {
            QString str= ui->textEdit->toPlainText();  //获取编辑区的内容
            file.write(str.toUtf8()); //获取完之后就写，并且转化成为utf8编码

        }
        file.close();
    }
}
