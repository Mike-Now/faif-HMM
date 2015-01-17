#include "MLReg.hpp"

#include <iostream>
#include <fstream>

#include <boost/archive/text_oarchive.hpp>

int main(){
	std::ofstream ofs("serial");

	boost::archive::text_oarchive oa(ofs);

	faif::ml::MLReg<faif::ValueNominal<std::string> > test;

	oa << test;

    std::cout<<"All tests for Multinomial Logistic Regression Classifier passed."<<std::endl;

    return 0;
}
