/** file with Discretizer tests */

#if defined(_MSC_VER) && (_MSC_VER >= 1400)
//msvc8.0 generuje osrzezenie dla TEST_TOOLS
#pragma warning(disable:4389)
//msvc8.0 generuje ostrzezenie gdy tylko sie rzuca wyjatek (jak w ExceptionsTest)
#pragma warning(disable:4702)
//msvc8.0 generuje smieci dla biblioteki boost::date_time
#pragma warning(disable:4512)
#pragma warning(disable:4127)
#pragma warning(disable:4100)
#pragma warning(disable:4245)
#pragma warning(disable:4701)
//msvc10.0 bad warning for boost::minmax
#pragma warning(disable:4913)
//msvc8.0 generuje smieci dla std::transform, std::generate_n, std::fill_n
#pragma warning(disable:4996)
#endif


#include <iostream>
#include <string>
#include <limits>
#include <sstream>
#include <iterator>
#include <algorithm>

#include <boost/bind.hpp>
#include <boost/ref.hpp>

#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include "../src/timeseries/Discretizer.hpp"

using namespace std;
using namespace boost;
using namespace faif::timeseries;
using boost::unit_test::test_suite;

BOOST_AUTO_TEST_SUITE( faif_discretizer_test )

BOOST_AUTO_TEST_CASE( Discretizer_vector_test )
{
	const double TAB[] = { 0.0, 10.0, 100.0, 99.0 };
	const int TAB_SIZE = sizeof(TAB) / sizeof(TAB[0]);
	std::vector<double> v(TAB, TAB + TAB_SIZE);
	Discretizer<double> s = createEqualWidthSections(v.begin(), v.end(), 4 );
	BOOST_REQUIRE_EQUAL( s.size(), 4U );
	BOOST_CHECK_CLOSE( s[0].getMin(), 0.0, 0.1 );
	BOOST_CHECK_CLOSE( s[0].getMax(), 25.0, 0.1 );
	BOOST_CHECK_CLOSE( s[1].getMin(), 25.0, 0.1 );
	BOOST_CHECK_CLOSE( s[1].getMax(), 50.0, 0.1 );
}

BOOST_AUTO_TEST_CASE( Discretizer_empty_test )
{
	std::vector<double> v;
	Discretizer<double> s = createEqualWidthSections(v.begin(), v.end(), 2 );
	BOOST_REQUIRE_EQUAL( s.size(), 0U );
}

BOOST_AUTO_TEST_CASE( Discretizer_sample_test )
{
	const double TAB[] = { 0.0, 10.0, 100.0, 99.0 };
	const int TAB_SIZE = sizeof(TAB) / sizeof(TAB[0]);
	std::vector<double> v(TAB, TAB + TAB_SIZE);
	Discretizer<double> s = createEqualWidthSections(v.begin(), v.end(), 2 );

	BOOST_CHECK_EQUAL( s.discretize( -1.0 ), 0);
	BOOST_CHECK_EQUAL( s.discretize( 0.0 ), 0);
	BOOST_CHECK_EQUAL( s.discretize( 49.0 ), 0);
	BOOST_CHECK_EQUAL( s.discretize( 50.0 ), 0);
	BOOST_CHECK_EQUAL( s.discretize( 51.0 ), 1);
	BOOST_CHECK_EQUAL( s.discretize( 99.0 ), 1);
	BOOST_CHECK_EQUAL( s.discretize( 100.0 ), 1);
	BOOST_CHECK_EQUAL( s.discretize( 101.0 ), 1);
	BOOST_CHECK_EQUAL( s.discretize( 1000.0 ), 1);
}

BOOST_AUTO_TEST_CASE( kSr_discretizer_vector_test )
{
	const double TAB[] = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0  };
	const int TAB_SIZE = sizeof(TAB) / sizeof(TAB[0]);
 	std::vector<double> v(TAB, TAB + TAB_SIZE);
	Discretizer<double> s = createKMeansSections(v.begin(), v.end(), 2 );
	BOOST_REQUIRE_EQUAL( s.size(), 2U );
	BOOST_CHECK_CLOSE( s[0].getMin(), 0.0, 10.0 );
	BOOST_CHECK_CLOSE( s[0].getMax(), 5.5, 10.0 );
	BOOST_CHECK_CLOSE( s[1].getMin(), 5.5, 10.0 );
	BOOST_CHECK_CLOSE( s[1].getMax(), 11.0, 10.0 );
}

BOOST_AUTO_TEST_CASE( kSr_discretizer__empty_test )
{
	std::vector<double> v;
	Discretizer<double> s = createKMeansSections( v.begin(), v.end(), 2 );
	BOOST_REQUIRE_EQUAL( s.size(), 0U );
}

BOOST_AUTO_TEST_CASE( kSr_discretizer_vector_test2 )
{
	const double TAB[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 40.0, 50.0, 60.0, 100.0 };
	const int TAB_SIZE = sizeof(TAB) / sizeof(TAB[0]);
	std::vector<double> v(TAB, TAB + TAB_SIZE);
	Discretizer<double> s = createKMeansSections(v.begin(), v.end(), 2 ) ;
	BOOST_REQUIRE_EQUAL( s.size(), 2U );

	BOOST_CHECK_CLOSE( s[0].getMin(), 0.0,   10.0 );
	BOOST_CHECK_CLOSE( s[0].getMax(), 31.25, 10.0 );
	BOOST_CHECK_CLOSE( s[1].getMin(), 31.25, 10.0 );
	BOOST_CHECK_CLOSE( s[1].getMax(), 100.0, 10.0 );
}

BOOST_AUTO_TEST_CASE( discretizer_serialize_test )
{
	const double TAB[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 40.0, 50.0, 60.0, 100.0 };
	const int TAB_SIZE = sizeof(TAB) / sizeof(TAB[0]);
	std::vector<double> v(TAB, TAB + TAB_SIZE);
	Discretizer<double> s = createKMeansSections(v.begin(), v.end(), 2 ) ;
	BOOST_REQUIRE_EQUAL( s.size(), 2U );

	BOOST_CHECK_CLOSE( s[0].getMin(), 0.0,   10.0 );
	BOOST_CHECK_CLOSE( s[0].getMax(), 31.25, 10.0 );
	BOOST_CHECK_CLOSE( s[1].getMin(), 31.25, 10.0 );
	BOOST_CHECK_CLOSE( s[1].getMax(), 100.0, 10.0 );

	ostringstream oss;
	boost::archive::text_oarchive oa(oss);
	oa << s;

	//cout << oss.str() << endl;

	Discretizer<double> s2;

    std::istringstream iss(oss.str());
    boost::archive::text_iarchive ia(iss);

	ia >> s2;

	BOOST_REQUIRE_EQUAL( s2.size(), 2U );

	BOOST_CHECK_CLOSE( s2[0].getMin(), 0.0,   10.0 );
	BOOST_CHECK_CLOSE( s2[0].getMax(), 31.25, 10.0 );
	BOOST_CHECK_CLOSE( s2[1].getMin(), 31.25, 10.0 );
	BOOST_CHECK_CLOSE( s2[1].getMax(), 100.0, 10.0 );
}



BOOST_AUTO_TEST_SUITE_END()
