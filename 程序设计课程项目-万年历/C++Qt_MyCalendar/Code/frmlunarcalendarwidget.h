#ifndef FRMLUNARCALENDARWIDGET_H
#define FRMLUNARCALENDARWIDGET_H


#include <QWidget>
#include <QPainter>

namespace Ui {
class frmLunarCalendarWidget;
}

class frmLunarCalendarWidget : public QWidget
{
    Q_OBJECT

public:
    explicit frmLunarCalendarWidget(QWidget *parent = 0);
    ~frmLunarCalendarWidget();

private:
    Ui::frmLunarCalendarWidget *ui;

private slots:
    void initForm();
    void on_cboxCalendarStyle_currentIndexChanged(int index);
    void on_cboxSelectType_currentIndexChanged(int index);
    void on_cboxWeekNameFormat_currentIndexChanged(int index);
    void on_ckShowLunar_stateChanged(int arg1);
    void on_pushButton_clicked();

    void on_pushButton_2_clicked();

protected:
    void paintEvent(QPaintEvent *);

};

#endif // FRMLUNARCALENDARWIDGET_H
