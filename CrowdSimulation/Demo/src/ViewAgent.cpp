///////////////////////////////////////////////////
// Low Level Parallel Programming 2017.
//
//     ==== There is no need to change this file ====
// 

#include "ViewAgent.h"
#include "MainWindow.h"
#include <QGraphicsItemAnimation>

// Memory leak check with msvc++
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#ifdef _DEBUG
#define new new(_NORMAL_BLOCK, __FILE__, __LINE__)
#endif

ViewAgent::ViewAgent(Ped::Tagent * agent, QGraphicsScene * scene) : agent(agent)
{
	QBrush greenBrush(Qt::green);
	QPen outlinePen(Qt::black);
	outlinePen.setWidth(2);

	rect = scene->addRect(MainWindow::cellToPixel(agent->getX()), MainWindow::cellToPixel(agent->getY()),
		MainWindow::cellsizePixel - 1, MainWindow::cellsizePixel - 1, outlinePen, greenBrush);
}

void ViewAgent::paint(QColor color){
	QBrush brush(color);
	rect->setBrush(brush);
	rect->setRect(MainWindow::cellToPixel(agent->getX()), MainWindow::cellToPixel(agent->getY()),
		MainWindow::cellsizePixel - 1, MainWindow::cellsizePixel - 1);
}

const std::pair<int, int> ViewAgent::getPosition(){
	return std::make_pair(agent->getX(), agent->getY());
}
