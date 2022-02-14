#-------------------------------------------------
#
# Project created by QtCreator 2017-01-05T22:11:54
#
#-------------------------------------------------

QT       += core gui
RC_FILE = icon.rc
RC_ICONS +=calendar.ico

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET      = lunarcalendarwidget
TEMPLATE    = app
DESTDIR     = $$PWD/../bin
CONFIG      += qt warn_off
RESOURCES   += main.qrc

SOURCES     += main.cpp
SOURCES     += frmlunarcalendarwidget.cpp
SOURCES     += lunarcalendaritem.cpp
SOURCES     += lunarcalendarinfo.cpp
SOURCES     += lunarcalendarwidget.cpp

HEADERS     += frmlunarcalendarwidget.h
HEADERS     += lunarcalendaritem.h
HEADERS     += lunarcalendarinfo.h
HEADERS     += lunarcalendarwidget.h

FORMS       += frmlunarcalendarwidget.ui

SUBDIRS += \
    C:/Users/PC_SKY_WYT/Downloads/feiyangqingyun-QWidgetDemo-master/QWidgetDemo/emailtool/emailtool.pro
