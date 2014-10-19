#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

#include <boost/serialization/nvp.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>

#include <faif/learning/NaiveBayesian.hpp>

using namespace std;
using namespace faif;
using namespace faif::ml;

int main() {


	NaiveBayesian nb;
	{
		ostringstream oss;
		boost::archive::xml_oarchive oa(oss);
		oa << boost::serialization::make_nvp("NBC", nb );
		//cout << oss.str();
	}

	std::ifstream ifile("x.xml");
	boost::archive::xml_iarchive ia(ifile);
	ia >> boost::serialization::make_nvp("NBC", nb); //de-serialize the stored object

	{
		ostringstream oss;
		boost::archive::xml_oarchive oa(oss);
		oa << boost::serialization::make_nvp("NBC", nb );
		cout << oss.str();
	}
	return 0;
}
