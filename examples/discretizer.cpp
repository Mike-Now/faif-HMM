#include <iostream>
#include <sstream>
#include <string>

#include <boost/serialization/nvp.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <faif/timeseries/Discretizer.hpp>

using namespace std;
using namespace faif;
using namespace faif::timeseries;

int main() {

	const double TAB[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 40.0, 50.0, 60.0, 100.0 };
	const int TAB_SIZE = sizeof(TAB) / sizeof(TAB[0]);
	std::vector<double> v(TAB, TAB + TAB_SIZE);

	/** two sections: <0.0, 31.25) and <31.25, 100) */
	Discretizer<double> s = createKMeansSections(v.begin(), v.end(), 2 ) ;

	ostringstream oss;
	boost::archive::text_oarchive oa(oss);
	oa << s;

	Discretizer<double> s2;

    std::istringstream iss(oss.str());
    boost::archive::text_iarchive ia(iss);
	ia >> s2;

	std::cout << (s2.discretize(0.0) == 0 )
			  << (s2.discretize(31.0) == 0 )
			  << (s2.discretize(32.0) == 1 )
			  << (s2.discretize(200.0) == 1 )
			  << std::endl;

	return 0;
}
