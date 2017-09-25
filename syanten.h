
#pragma once
#ifndef __SYANTEN
#define __SYANTEN

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <string>
#include <array>
#include <boost/python.hpp>
using namespace std;

class Syanten{

	void mentu_cut(int);
	void taatu_cut(int);

	std::array<size_t,39> tehai;
	int mentu;
	int toitu;
	int kouho;
	int fuurosuu;
	int temp;
	int syanten_normal;
	int checkNormalSyanten();
	map<int, vector<int>> map_; 

public:
	Syanten();
	int NormalSyanten();
	int KokusiSyanten();
	int TiitoituSyanten();

	bool checkAtama();

	void set_tehai(boost::python::list t);
	void set_fuurosuu(int a){fuurosuu=a;}
	void clear();
};

#endif
