/** plik zawiera test klas utils */

#if defined(_MSC_VER) && (_MSC_VER >= 1400)
//msvc8.0 generuje warning dla boost::numeric::matrix
#pragma warning(disable:4127)
#pragma warning(disable:4512)
#pragma warning(disable:4996)
//msvc9.0 generuje smieci dla boost/date_time
#pragma warning(disable:4244)
//msvc10.0 generuje smieci dla boost/math/distributions
#pragma warning(disable:4702)

#endif

#include <iostream>

#include <string>
#include <sstream>

#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/blas.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <boost/thread.hpp>

#include "../src/utils/Random.hpp"
#include "../src/utils/RandomCustomDistr.hpp"
#include "../src/utils/GaussEliminator.h"
#include "../src/utils/Power.hpp"


using namespace std;
using namespace faif;
using namespace boost;
using namespace boost::numeric;
using boost::unit_test::test_suite;

BOOST_AUTO_TEST_SUITE( faif_utils_test )

BOOST_AUTO_TEST_CASE( RandomGeneratorTest ) {

    const int NUM = 1000;

	RandomDouble r;
    double s = 0.0;
    for(int i=0;i<NUM;++i) {
		s += r();
    }

	RandomDouble rr;
	double ss = 0.0;
    for(int i=0;i<NUM;++i) {
		ss += rr();
    }

	BOOST_CHECK( s/NUM > 0.4 && s/NUM < 0.6 );
	BOOST_CHECK( ss/NUM > 0.4 && ss/NUM < 0.6 );
	BOOST_CHECK( s != ss );

	RandomDouble u(-1.0,1.0);
	double mu = 0.0;
	for(int i=0;i<NUM;++i)
		mu += u();

	BOOST_CHECK( mu/NUM > -0.1 && mu/NUM < 0.1 );

	const int MAX = 10;
	RandomInt ri(0,MAX);
	for(int j=0;j<NUM;j++) {
		BOOST_CHECK( ri() >= 0 );
		BOOST_CHECK( ri() <= MAX );
	}

	RandomInt rj(0,0);
	for(int j = 0; j < NUM; j++) {
		BOOST_CHECK( rj() == 0 );
	}

}

BOOST_AUTO_TEST_CASE( RandomNormalDistributionTest ) {

    const int NUM = 1000;

	RandomNormal n(0.0, 1.0);
	double s = 0.0;
    for(int i=0;i<NUM;++i) {
		s += n();
		//cout << n() << endl;
    }
// 	cout << s << endl;
// 	cout << s/NUM << endl;
	BOOST_CHECK( s/NUM >= -0.1 && s/NUM <= 0.1 );

}


namespace {
	//watek, ktory korzysta z generatora liczb losowych
	class ThreadRand {
	public:
		ThreadRand(int num) : num_(num) { }
		//bada kolejne liczby calkowite i zlicza liczby pierwsze.
		void operator()() {
			for(int i=0;i<num_;++i)
				rand_();
		}
	private:
		int num_;
		RandomDouble rand_;
	};
}

BOOST_AUTO_TEST_CASE( RandomMultithreadTest ) {
	const int NUM = 10000;
	const int NUM_THREADS = 10;
	ThreadRand r(NUM);
    boost::thread_group threads;
    for (int i = 0; i < NUM_THREADS; ++i)
        threads.create_thread(r);
    threads.join_all();
}


BOOST_AUTO_TEST_CASE( general_distribution_values_test )
{
    DistrValue val1(1.0, 3.0, 0.5);
    DistrValue val2(3.0, 1.0, 0.5);
    BOOST_CHECK_EQUAL( val1, val2 );

    DistrValues v1;
    v1.push_back( DistrValue(1.0, 2.0, 0.2 ) );
    v1.push_back( DistrValue(0.0, 1.0, 0.2 ) );
    DistrValues v1_out =  makeProbabilityDensity(v1);
    BOOST_REQUIRE_EQUAL( v1_out.size(), 2U );
    BOOST_CHECK( v1_out[0] == DistrValue( 0.0, 1.0, 0.5 ) );
    BOOST_CHECK( v1_out[1] == DistrValue( 1.0, 2.0, 0.5 ) );

    DistrValues v2;
    v2.push_back( DistrValue(0.0, 1.0, 0.2 ) );
    v2.push_back( DistrValue(0.1, 0.9, 0.2 ) );
    DistrValues v2_out =  makeProbabilityDensity(v2);
    BOOST_REQUIRE_EQUAL( v2_out.size(), 1U );
    BOOST_CHECK( v2_out[0] == DistrValue( 0.0, 1.0, 1.0 ) );

    DistrValues v3;
    v3.push_back( DistrValue(0.0, 1.0, 0.2 ) );
    v3.push_back( DistrValue(0.1, 2.0, 0.2 ) );
    DistrValues v3_out =  makeProbabilityDensity(v3);
    BOOST_REQUIRE_EQUAL( v3_out.size(), 2U );
    BOOST_CHECK( v3_out[0] == DistrValue( 0.0, 1.0, 0.5 ) );
    BOOST_CHECK( v3_out[1] == DistrValue( 1.0, 2.0, 0.5 ) );

    DistrValues v4;
    v4.push_back( DistrValue(0.0, 1.0, 0.2 ) );
    v4.push_back( DistrValue(2.0, 3.0, 0.2 ) );
    DistrValues v4_out =  makeProbabilityDensity(v4);
    //cout << v4_out << endl;
    BOOST_REQUIRE_EQUAL( v4_out.size(), 3U );
    BOOST_CHECK( v4_out[0] == DistrValue( 0.0, 1.0, 0.5 ) );
    BOOST_CHECK( v4_out[1] == DistrValue( 1.0, 2.0, 0.0 ) );
    BOOST_CHECK( v4_out[2] == DistrValue( 2.0, 3.0, 0.5 ) );
}

BOOST_AUTO_TEST_CASE( random_custom_distr_test ) {

#if defined(_MSC_VER) && (_MSC_VER >= 1400) && defined(_DEBUG)
    cerr << "not tested - error in debug version of std::lower_bound implemented in msvc" << endl;
#else
    DistrValues v;
    v.push_back( DistrValue( -0.5, 0.5, 1.0 ) );
    RandomCustomDistr h(v);

    BOOST_CHECK_CLOSE( h.getMean(), 0.0, 1.0 );
    BOOST_CHECK_CLOSE( h.getStandardDeviation(), 0.29, 1.0 );

    BOOST_CHECK_CLOSE( h.getProbabilityDensity( -1.), 0.0, 1.0 );
    BOOST_CHECK_CLOSE( h.getProbabilityDensity( -.5), 1.0, 1.0 );
    BOOST_CHECK_CLOSE( h.getProbabilityDensity( 0.0), 1.0, 1.0 );
    BOOST_CHECK_CLOSE( h.getProbabilityDensity( 1.0), 0.0, 1.0 );
    BOOST_CHECK_CLOSE( h.getProbabilityDensity( 1.5), 0.0, 1.0 );

    BOOST_CHECK_CLOSE( h.getDistribution( 0.0), 0.5, 1.0 );
    BOOST_CHECK_CLOSE( h.getDistribution( 0.5), 1.0, 1.0 );
    BOOST_CHECK_CLOSE( h.getDistribution( 1.0), 1.0, 1.0 );

    BOOST_CHECK_CLOSE( h.getQuantile(0.9),  0.4, 1.0 );
    BOOST_CHECK_CLOSE( h.getQuantile(0.95), 0.45, 1.0 );

#endif
}

BOOST_AUTO_TEST_CASE( random_custom_distr_test2 ) {

#if defined(_MSC_VER) && (_MSC_VER >= 1400) && defined(_DEBUG)
    cerr << "not tested - error in debug version of std::lower_bound implemented in msvc" << endl;
#else

    DistrValues v;
    v.push_back( DistrValue(0.0, 1.0, 0.2 ) );
    v.push_back( DistrValue(2.0, 3.0, 0.2 ) );

    RandomCustomDistr h(v);
    BOOST_CHECK_CLOSE( h.getMean(), 1.5, 1.0);

    //variation: Integral from 0 to 1 of (x-1.5)^2 dx + Integral from 2 to 3 of (x-1.5)^2 dx = 13/12
    BOOST_CHECK_CLOSE( h.getStandardDeviation(), sqrt(13.0/12.0), 1.0 );

    BOOST_CHECK_CLOSE( h.getProbabilityDensity( -.5), 0.0,  1.0 );
    BOOST_CHECK_CLOSE( h.getProbabilityDensity( 0.0), 0.5, 1.0 );
    BOOST_CHECK_CLOSE( h.getProbabilityDensity( 0.5), 0.5, 1.0 );
    BOOST_CHECK_CLOSE( h.getProbabilityDensity( 1.0), 0.0, 1.0 );
    BOOST_CHECK_CLOSE( h.getProbabilityDensity( 1.5), 0.0, 1.0 );
    BOOST_CHECK_CLOSE( h.getProbabilityDensity( 2.0), 0.5, 1.0 );
    BOOST_CHECK_CLOSE( h.getProbabilityDensity( 2.5), 0.5, 1.0 );
    BOOST_CHECK_CLOSE( h.getProbabilityDensity( 3.5), 0.0, 1.0 );

    BOOST_CHECK_CLOSE( h.getDistribution( -.5), 0.0, 1.0 );
    BOOST_CHECK_CLOSE( h.getDistribution( 0.0), 0.0, 1.0 );
    BOOST_CHECK_CLOSE( h.getDistribution( 0.5), 0.25, 1.0 );
    BOOST_CHECK_CLOSE( h.getDistribution( 0.5), 0.25, 1.0 );
    BOOST_CHECK_CLOSE( h.getDistribution( 1.0), 0.5, 1.0 );
    BOOST_CHECK_CLOSE( h.getDistribution( 1.5), 0.5, 1.0 );
    BOOST_CHECK_CLOSE( h.getDistribution( 2.0), 0.5, 1.0 );
    BOOST_CHECK_CLOSE( h.getDistribution( 2.5), 0.75, 1.0 );
    BOOST_CHECK_CLOSE( h.getDistribution( 3.0), 1.0, 1.0 );
    BOOST_CHECK_CLOSE( h.getDistribution( 3.5), 1.0, 1.0 );

    BOOST_CHECK_CLOSE( h.getQuantile(0.0),  0.0, 1.0 );
    BOOST_CHECK_CLOSE( h.getQuantile(0.1),  0.2, 1.0 );
    BOOST_CHECK_CLOSE( h.getQuantile(0.5),  2.0, 1.0 );
    BOOST_CHECK_CLOSE( h.getQuantile(0.9),  2.8, 1.0 );
    BOOST_CHECK_CLOSE( h.getQuantile(1.0),  3.0, 1.0 );
#endif
}

BOOST_AUTO_TEST_CASE( random_custom_creator_test ) {
	RandomCustomCreator creator(0.5);
	creator.addValue(0.0);
	creator.addValue(0.2);
	creator.addValue(0.4);
	creator.addValue(0.6);
	creator.addValue(0.8);
	creator.addValue(1.0);
	creator.addValue(1.2);
	creator.addValue(1.4);
	creator.addValue(1.6);
	creator.addValue(1.8);
	BOOST_CHECK_EQUAL( creator.getNumValues(), 10 );
    RandomCustomDistr h = creator.getRandomCustomDistr();
	BOOST_CHECK_EQUAL( h.getValues().size(), 4U );
}


BOOST_AUTO_TEST_CASE( dist_calculate_epsilon_test ) {
	BOOST_CHECK_CLOSE(RandomCustomCreator::calculateEpsilon(1.0, 1) ,         112.8, 1.0);
	BOOST_CHECK_CLOSE(RandomCustomCreator::calculateEpsilon(1.0, 10) ,         35.68, 1.0);
	BOOST_CHECK_CLOSE(RandomCustomCreator::calculateEpsilon(1.0, 100) ,        11.28, 1.0);
	BOOST_CHECK_CLOSE(RandomCustomCreator::calculateEpsilon(1.0, 1000) ,        3.568, 1.0);
	BOOST_CHECK_CLOSE(RandomCustomCreator::calculateEpsilon(1.0, 10000) ,       1.128, 1.0);
	BOOST_CHECK_CLOSE(RandomCustomCreator::calculateEpsilon(1.0, 100000) ,      0.3568, 1.0);
	BOOST_CHECK_CLOSE(RandomCustomCreator::calculateEpsilon(1.0, 1000000) ,     0.1128, 1.0);
	BOOST_CHECK_CLOSE(RandomCustomCreator::calculateEpsilon(1.0, 10000000) ,    0.03568, 1.0);
}


BOOST_AUTO_TEST_CASE( GaussEliminatorTest ) {

	const int n1 = 20;
	ublas::matrix<double> m1(n1,n1);
	m1 = ublas::identity_matrix<double>(n1);
	ublas::vector<double> y1(n1);
	for(int i = 0; i < n1; ++i )
		y1(i) = i;
	ublas::vector<double> res1 = GaussEliminator(m1,y1);
	for(int i = 0; i < n1; ++i )
		BOOST_CHECK_CLOSE( y1(i), res1(i), 0.1 );

	const int n = 3;
    ublas::matrix<double> m(n,n);
    m(0, 0) = 2; m(0, 1) = 1; m(0, 2) = 3;
    m(1, 0) = 1; m(1, 1) = 3; m(1, 2) = 4;
    m(2, 0) = 3; m(2, 1) = 4; m(2, 2) = 5;
    ublas::vector<double> x(n);
    x(0) = 4;
    x(1) = 5;
    x(2) = 7;
	//nie niszczy m oraz x (ale wolniejsze, bo robi kopie)
	ublas::vector<double> y = GaussEliminator(m,x);

	BOOST_CHECK_CLOSE( y(0), 0.4, 0.1 );
	BOOST_CHECK_CLOSE( y(1), 0.2, 0.1 );
	BOOST_CHECK_CLOSE( y(2), 1.0, 0.1 );
	//niszczy macierz m oraz x
	ublas::vector<double> yy = GaussEliminatorRef(m,x);

	BOOST_CHECK_CLOSE( yy(0), 0.4, 0.1 );
	BOOST_CHECK_CLOSE( yy(1), 0.2, 0.1 );
	BOOST_CHECK_CLOSE( yy(2), 1.0, 0.1 );

}

BOOST_AUTO_TEST_CASE( IntPowerTest ) {

	BOOST_CHECK_CLOSE( int_power<3>(3.0), 27.0, 0.1 );
	BOOST_CHECK_CLOSE( int_power<4>(3.0), 81.0, 0.1 );
	BOOST_CHECK_CLOSE( int_power<1000>(1.1), 2.47e+41, 0.1 );
}

BOOST_AUTO_TEST_SUITE_END()
