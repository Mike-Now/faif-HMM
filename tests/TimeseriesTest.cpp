/** plik zawiera test klas timeseries */

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

#include <boost/date_time/gregorian/gregorian.hpp>

#include "../src/timeseries/TimeSeries.hpp"
#include "../src/timeseries/Transformations.hpp"
#include "../src/timeseries/Discretizer.hpp"

using namespace std;
using namespace boost;
using namespace faif::timeseries;
using boost::unit_test::test_suite;

BOOST_AUTO_TEST_SUITE( faif_timeseries_test )

BOOST_AUTO_TEST_CASE( BoostDateTimeTest ) {

    gregorian::date d(2007,gregorian::Jan,21);

    posix_time::ptime t1(d,posix_time::time_duration(12,5,1) );
    BOOST_CHECK_EQUAL( t1.date(), d );

    string t1str = posix_time::to_simple_string(t1);
    BOOST_CHECK_EQUAL(t1str,string("2007-Jan-21 12:05:01") );

    RealTime t2(d, posix_time::hours(12)+posix_time::minutes(10)+posix_time::seconds(4) );
    RealDuration td = t2 - t1;
    string tdstr = posix_time::to_simple_string(td);
    BOOST_CHECK_EQUAL(tdstr,string("00:05:03") );
}

namespace {
    struct RawDataStruct { const char* s; int v; };

    //TODO: moze uzyc z biblioteki?
    /** pomocniczy funktor - do konwersji wlasnych formatow zapisu daty */
    struct RealTimeFromStrFunctor : boost::noncopyable {

        stringstream ss_;

        RealTimeFromStrFunctor() : ss_("") {
            posix_time::time_input_facet* my_facet = new posix_time::time_input_facet("%d.%m.%Y %H:%M");
            ss_.imbue( std::locale( std::locale::classic(), my_facet) );
        }
        TimeValueReal operator()(const RawDataStruct& raw) {
            ss_.str(string(raw.s));
            RealTime t;
            ss_ >> t;
            return TimeValueReal(t,raw.v);
        }
    };
}


BOOST_AUTO_TEST_CASE( RawDataTest ) {
    // przykladowe dane (RAW_DATA)
    const RawDataStruct RAW_DATA_TAB[] = { {"01.02.1995 00:00",295},{"01.02.1995 01:00",267},{"01.02.1995 02:00",309},
                                           {"01.02.1995 03:00",281},{"01.02.1995 04:00",337},{"01.02.1995 05:00",597} };
    const int RAW_DATA_TAB_SIZE = sizeof(RAW_DATA_TAB)/sizeof(RAW_DATA_TAB[0]);

    RealTimeFromStrFunctor my_functor;
    TimeSeriesReal data;
    for(int i=0;i<RAW_DATA_TAB_SIZE;++i)
        data.push_back( my_functor(RAW_DATA_TAB[i]) );

    BOOST_CHECK_EQUAL(data.size(),RAW_DATA_TAB_SIZE);
}

BOOST_AUTO_TEST_CASE( TimeSeriesRealTest ) {

    RealTime t(gregorian::date(2007,gregorian::Jan,23), posix_time::hours(12)+posix_time::minutes(55) );
    Value v = 0.0;
    TimeValueReal tval(t,v);
    TimeValueReal tval2(t,v,0.0);
    BOOST_CHECK_EQUAL(tval, tval2);
    ostringstream os;
    os << tval;
    BOOST_CHECK_EQUAL(os.str(),string("(2007-Jan-23 12:55:00,0)") );

    //sprawdza konwersje z time_t
    long time_posix = 0x48000000;
    RealTime t2 = posix_time::from_time_t (time_posix );
    long time_posix2 = to_time_t(t2);

    BOOST_CHECK_EQUAL( time_posix, time_posix2);
}

namespace {
    //generator kolejnych timestamp-ow
    struct TimestampGenerator {
        TimestampGenerator(RealTime t, RealDuration d)
            : curr_(t), delta_(d) {}

        //generuje kolejna probke czasu
        long operator()() {
            long out = to_time_t(curr_);
            curr_ += delta_;
            return out;
        }
        RealTime curr_;
        RealDuration delta_;
    };
} //namespace

BOOST_AUTO_TEST_CASE( TimeSeriesRealCreateTest ) {

    const double VALUES[] = { 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8 };
    const int SIZE = sizeof(VALUES)/sizeof(VALUES[0]);

    long timestamps[SIZE];

    RealTime t(gregorian::date(2008,gregorian::Feb,13), posix_time::hours(16) );
    RealDuration d( posix_time::hours(1) );

    TimestampGenerator gen(t,d);
    generate_n( timestamps, SIZE, gen );

    double qualities[SIZE];
    fill_n( qualities, SIZE, 0.0 );

    TimeSeriesReal ts( timestamps, timestamps + SIZE, VALUES );

    BOOST_CHECK_EQUAL( ts.front().getTime(), t );
    BOOST_CHECK_EQUAL( ts.back().getTime(), t + d*(SIZE-1) );

    TimeSeriesReal ts2( t, d, VALUES, VALUES + SIZE);
    BOOST_CHECK_EQUAL_COLLECTIONS( ts.begin(), ts.end(), ts2.begin(), ts2.end() );
}

BOOST_AUTO_TEST_CASE( TSRealMethodsTest ) {
    const double VALUES[] = { 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8 };
    const int SIZE = sizeof(VALUES)/sizeof(VALUES[0]);

    RealTime t(gregorian::date(2008,gregorian::May,26), posix_time::hours(16));
    RealDuration d(posix_time::hours(1) );

    TimeSeriesReal ts( t, d, VALUES, VALUES + SIZE );
    BOOST_CHECK_CLOSE( ts.getSum(), 25.4, 0.5 );
    BOOST_CHECK_CLOSE( ts.getAvg(), 3.63, 0.5 );
    BOOST_CHECK_CLOSE( ts.getIntegral(), 68040.0, 0.5 );

    //empty timeseries
    TimeSeriesReal ts2(t, d, VALUES, VALUES);
    cout << ts2 << endl;
    BOOST_CHECK_CLOSE( ts2.getSum(), 0.0, 0.1 );
    BOOST_CHECK_CLOSE( ts2.getAvg(), 0.0, 0.1 );
    BOOST_CHECK_CLOSE( ts2.getIntegral(), 0.0, 0.1 );

    //timeseries with one timestamp
    TimeSeriesReal ts3(t, d, VALUES, VALUES + 1);
    BOOST_CHECK_CLOSE( ts3.getSum(), 0.2, 0.1 );
    BOOST_CHECK_CLOSE( ts3.getAvg(), 0.2, 0.1 );
    BOOST_CHECK_CLOSE( ts3.getIntegral(), 0.0, 0.1 );

}


BOOST_AUTO_TEST_CASE( TimeSeriesDigitTest ) {
    DigitTime t = 0;
    Value v = 0.0;
    Quality q = 0.5;
    TimeValueDigit tval(t,v,q);
    TimeValueDigit tval2 = tval;
    BOOST_CHECK( tval2 == tval );

    const TimeValueDigit TAB[] = { TimeValueDigit(0,0.0,0.5), TimeValueDigit(1,0.1,0.5), TimeValueDigit(2,0.2,0.5),
                                   TimeValueDigit(3,0.3,0.5), TimeValueDigit(4,0.4,0.5), TimeValueDigit(5,0.5,0.5) };
    const int TAB_SIZE = sizeof(TAB)/sizeof(TAB[0]);
    TimeSeriesDigit tseries(TAB, TAB + TAB_SIZE);
    BOOST_CHECK( static_cast<int>(tseries.size()) == TAB_SIZE );

    const double VALUES[] = { 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8 };
    const int VALUES_SIZE = sizeof(VALUES)/sizeof(VALUES[0]);
    TimeSeriesDigit tseries2( VALUES, VALUES + VALUES_SIZE);
    BOOST_CHECK( static_cast<int>(tseries2.size()) == VALUES_SIZE );

    TimeSeriesDigit tseries3(tseries2, -5 );
    BOOST_CHECK_EQUAL( tseries3[0], TimeValueDigit(-5,0.1) );

    BOOST_CHECK_CLOSE( tseries3.getSum(), 25.5, 0.1 );
    BOOST_CHECK_CLOSE( tseries3.getSumSquared(), 218.45, 0.1 );
    BOOST_CHECK_CLOSE( tseries3.getAvg(), 3.1875, 0.1 );
    BOOST_CHECK_CLOSE( tseries3.getSigmaSquared(), 1097.35, 0.1 );
    BOOST_CHECK_CLOSE( tseries3.getSigma(), 4.14, 0.1 );

    BOOST_CHECK_CLOSE( getAvgAbsDiff( tseries, tseries2 ), 3.0, 0.1 );
    BOOST_CHECK_CLOSE( getAvgAbsDiff( tseries, tseries2 ), 3.0, 0.1 );

    BOOST_CHECK_CLOSE( getAvgAbsDiff( tseries, tseries ), 0.0, 0.1 );
    BOOST_CHECK_CLOSE( getAvgAbsDiff( tseries2, tseries2 ), 0.0, 0.1 );

    BOOST_CHECK_CLOSE( getAvgAbsDiff( tseries3, tseries2 ), 49.6 / 13, 0.1 );
    BOOST_CHECK_CLOSE( getAvgAbsDiff( tseries2, tseries3 ), 49.6 / 13, 0.1 );

    BOOST_CHECK_CLOSE( getAvgRelDiff(tseries, tseries), 0.0, 0.1 );
    BOOST_CHECK_CLOSE( getAvgRelDiff(tseries, tseries2), 1.113, 0.5 );
    BOOST_CHECK_CLOSE( getAvgRelDiff(tseries2, tseries3), 1.203, 0.5 );

    TimeSeriesDigit tseries4( tseries3.begin(), tseries3.begin() + 3 );
    BOOST_CHECK_EQUAL( tseries4.size(), 3 );
    BOOST_CHECK_CLOSE( tseries4.getSum(), 0.7, 0.1 );


    //empty timeseries
    TimeSeriesDigit tseries5;
    BOOST_CHECK( tseries5.empty() );
    BOOST_CHECK_CLOSE( tseries5.getSum(), 0.0 , 0.1 );
    BOOST_CHECK_CLOSE( tseries5.getAvg(), 0.0, 0.1 );
    BOOST_CHECK_CLOSE( tseries5.getSigmaSquared(), 0.0, 0.1 );
    BOOST_CHECK_CLOSE( tseries5.getSigma(), 0.0, 0.1 );


}

BOOST_AUTO_TEST_CASE( EpsilonNaNTest ) {

    double epsilon = std::numeric_limits<double>::epsilon();
    double nan = std::numeric_limits<double>::quiet_NaN();

    const double V[] = { 0.0, epsilon, 1.0 };
    const int V_SIZE = sizeof(V)/sizeof(V[0]);
    TimeSeriesDigit tseries( V, V + V_SIZE);

    const double V2[] = { 1.0, epsilon, 0.0 };
    const int V2_SIZE = sizeof(V2)/sizeof(V2[0]);
    TimeSeriesDigit tseries2(V2, V2 + V2_SIZE );

    BOOST_CHECK_CLOSE( getAvgRelDiff(tseries, tseries2), 2.0, 0.5 );

    const double V3[] = { nan, 0.0, 0.0 };
    const int V3_SIZE = sizeof(V3)/sizeof(V3[0]);
    TimeSeriesDigit tseries3(V3, V3 + V3_SIZE );

    double out_nan = getAvgAbsDiff(tseries, tseries3);
    BOOST_CHECK( out_nan != out_nan );
    double out_nan2 = getAvgRelDiff(tseries, tseries3);
    BOOST_CHECK( out_nan2 != out_nan2 );
}


BOOST_AUTO_TEST_CASE( CorrelationTest ) {
    const double VALUES[] = { 1, 2, 0, 1, 1, 0, 1, 2, 1,
                              1, 2, 0, 1, 1, 0, 1, 2, 1,
                              1, 2, 0, 1, 1, 0, 1, 2, 1,
                              1, 2, 0, 1, 1, 0, 1, 2, 1,
                              1, 2, 0, 1, 1, 0, 1, 2, 1,
                              1, 2, 0, 1, 1, 0, 1, 2, 1,
                              1, 2, 0, 1, 1, 0, 1, 2, 1,
                              1, 2, 0, 1, 1, 0, 1, 2, 1,
                              1, 2, 0, 1, 1, 0, 1, 2, 1,
                              1, 2, 0, 1, 1, 0, 1, 2, 1 };
    const int VALUES_SIZE = sizeof(VALUES)/sizeof(VALUES[0]);
    TimeSeriesDigit ts( VALUES, VALUES + VALUES_SIZE);

    const double COREL[] = { 1.011, -026, -0.31, 0.47, -0.50, -0.51, 0.54, -0.19, -0.25, 1.01, -0.26 };
    const int COREL_SIZE = sizeof(COREL)/sizeof(COREL[0]);
    TimeSeriesDigit corel_ref( COREL, COREL + COREL_SIZE );

    TimeSeriesDigit cor = correlation(ts, ts, COREL_SIZE );
    //  BOOST_CHECK_EQUAL_COLLECTIONS( cor.begin(), cor.end(), corel_ref.begin(), corel_ref.end() );

    const double AUTOCOR[] = {1.46, 0.90, 0.87, 1.22, 0.79, 0.79, 1.25, 0.93, 0.90, 1.46, 0.89};
    const int AUTOCOR_SIZE = sizeof(AUTOCOR)/sizeof(AUTOCOR[0]);
    TimeSeriesDigit autocorel_ref(AUTOCOR, AUTOCOR + AUTOCOR_SIZE );

    TimeSeriesDigit autocor = ts.autoCorrelationE(AUTOCOR_SIZE);
    //  BOOST_CHECK_EQUAL_COLLECTIONS( autocor.begin(), autocor.end(), autocorel_ref.begin(), autocorel_ref.end() );
}

BOOST_AUTO_TEST_CASE( TransformationTest ) {
    gregorian::date d(2008,gregorian::Sep,5);
    RealTime present( d, posix_time::hours(12) );
    RealDuration delta( posix_time::hours(1) );

    Transformation trans(present, delta);

    BOOST_CHECK_EQUAL( trans.toReal(0), present );
    BOOST_CHECK_EQUAL( trans.toReal(1), RealTime(d, posix_time::hours(13) ) );
    BOOST_CHECK_EQUAL( trans.toReal(-1), RealTime(d, posix_time::hours(11) ) );
    BOOST_CHECK_EQUAL( trans.toReal(-12), RealTime(d, posix_time::hours(0) ) );
    BOOST_CHECK_EQUAL( trans.toReal(+11), RealTime(d, posix_time::hours(23) ) );

    BOOST_CHECK_EQUAL( trans.toDigit(present), 0 );
    BOOST_CHECK_EQUAL( trans.toDigit( RealTime(d, posix_time::hours(13) ) ), 1 );
    BOOST_CHECK_EQUAL( trans.toDigit( RealTime(d, posix_time::hours(11) ) ), -1 );

    //test range borders
    BOOST_CHECK_EQUAL( trans.toDigit( RealTime(d, posix_time::hours(11) + posix_time::minutes(29) ) ), -1 );
    BOOST_CHECK_EQUAL( trans.toDigit( RealTime(d, posix_time::hours(11) + posix_time::minutes(30) ) ), 0 );
    BOOST_CHECK_EQUAL( trans.toDigit( RealTime(d, posix_time::hours(12) + posix_time::minutes(29) ) ), 0 );
    BOOST_CHECK_EQUAL( trans.toDigit( RealTime(d, posix_time::hours(12) + posix_time::minutes(30) ) ), 1 );

    BOOST_CHECK_EQUAL( trans.toDigit( RealTime(d, posix_time::time_duration(10,29,59 ) ) ), -2 );
    BOOST_CHECK_EQUAL( trans.toDigit( RealTime(d, posix_time::time_duration(10,30,00 ) ) ), -1 );

    BOOST_CHECK_EQUAL( trans.toDigit( RealTime(d, posix_time::time_duration(13,29,59 ) ) ), 1 );
    BOOST_CHECK_EQUAL( trans.toDigit( RealTime(d, posix_time::time_duration(13,30,00 ) ) ), 2 );
}

BOOST_AUTO_TEST_CASE( TransformationMillisecTest ) {
    gregorian::date d(2010,gregorian::Aug, 3);
    RealTime present( d, posix_time::hours(12) );
    RealDuration delta( posix_time::milliseconds(10) );

    Transformation trans(present, delta);

    BOOST_CHECK_EQUAL( trans.toReal(0), present );
    BOOST_CHECK_EQUAL( trans.toReal(1),  RealTime(present + posix_time::millisec(10) ) );
    BOOST_CHECK_EQUAL( trans.toReal(-1), RealTime(present - posix_time::millisec(10) ) );

    BOOST_CHECK_EQUAL( trans.toDigit(present), 0 );
    BOOST_CHECK_EQUAL( trans.toDigit(present + posix_time::millisec(10) ), 1 );
    BOOST_CHECK_EQUAL( trans.toDigit(present - posix_time::millisec(10) ), -1);
}


BOOST_AUTO_TEST_CASE( LinearResamplingTest ) {
    //generuje szereg czasowy
    const int SIZE = 20;
    gregorian::date today = gregorian::date(2008,gregorian::Sep, 8);
    RealTime t(today, posix_time::hours(9) );
    RealDuration d( posix_time::hours(1) );
    TimeSeriesReal ts_real;
    for(int i=0;i<SIZE;++i) {
        ts_real.push_back( TimeValueReal(t, i, 0.0) );
        t += d;
    }

    // cout << ts_real << endl;

    RealTime present(today, posix_time::hours(12) );
    RealDuration delta( posix_time::hours(1) );

    Transformation trans(present, delta);

    TimeSeriesDigit ts = create(ts_real, trans);

    BOOST_CHECK_EQUAL( ts.size(), SIZE );
    BOOST_CHECK_EQUAL( ts.front().getTime(), -3);
    BOOST_CHECK_EQUAL( ts.front().getValue(), 0);
    BOOST_CHECK_EQUAL( ts.back().getTime(), 16);
    BOOST_CHECK_EQUAL( ts.back().getValue(), 19);

	//agregacja (dla pelnych danych, zeby sprawdzic poprawnosc)
	TimeSeriesDigit ts_agr = create(ts_real, Transformation(present, delta*2) );
	//cout << ts << endl;
	BOOST_CHECK_EQUAL( ts_agr.size(), (SIZE-1)/2+1);
	BOOST_CHECK_CLOSE( ts_agr[0].getValue(), 0.5, 1.0);
	BOOST_CHECK_CLOSE( ts_agr[1].getValue(), 2.5, 1.0);
	BOOST_CHECK_CLOSE( ts_agr[2].getValue(), 4.5, 1.0);
	BOOST_CHECK_CLOSE( ts_agr[3].getValue(), 6.5, 1.0);
	BOOST_CHECK_CLOSE( ts_agr[4].getValue(), 8.5, 1.0);
	BOOST_CHECK_CLOSE( ts_agr[5].getValue(),10.5, 1.0);

	// tutaj usuwa troche sampli (robi ,,dziury'')
	ts_real.erase( ts_real.begin() + 9 ); //dziura na pozycji 9
	ts_real.erase( ts_real.begin() + 5 ); //dziura na pozycji 5
	ts_real.erase( ts_real.begin() + 4 ); //dziura na pozycji 4

 	TimeSeriesDigit ts2 = create(ts_real, trans);
// 	//   cout << ts << endl;
// 	//   cout << ts2 << endl;
	//powinien odtworzyc calkowicie
	BOOST_CHECK_EQUAL_COLLECTIONS( ts.begin(), ts.end(), ts2.begin(), ts2.end() );

	//tutaj agreguje
	trans = Transformation(present, delta*3);
	ts = create(ts_real, trans);

	BOOST_CHECK_EQUAL(ts.size(), (SIZE-1)/3 + 1);

	//turaj resampluje
	trans = Transformation(present, delta/4);
	ts = create(ts_real, trans);
	BOOST_CHECK_EQUAL( ts.size(), 4*(SIZE-1) + 1);
	BOOST_CHECK_CLOSE( ts[0].getValue(), 0.0,  1.0 );
	BOOST_CHECK_CLOSE( ts[1].getValue(), 0.25, 1.0 );
	BOOST_CHECK_CLOSE( ts[2].getValue(), 0.5,  1.0 );
	BOOST_CHECK_CLOSE( ts[3].getValue(), 0.75, 1.0 );
	BOOST_CHECK_CLOSE( ts[4].getValue(), 1.0,  1.0 );
	BOOST_CHECK_CLOSE( ts[5].getValue(), 1.25, 1.0 );

}

BOOST_AUTO_TEST_CASE( CreateRealTSTest ) {

    const double VALUES[] = { 1, 2, 0, 1, 1, 0, 1, 2, 1,
                              1, 2, 0, 1, 1, 0, 1, 2, 1,
                              1, 2, 0, 1, 1, 0, 1, 2, 1,
                              1, 2, 0, 1, 1, 0, 1, 2, 1,
                              1, 2, 0, 1, 1, 0, 1, 2, 1,
                              1, 2, 0, 1, 1, 0, 1, 2, 1,
                              1, 2, 0, 1, 1, 0, 1, 2, 1,
                              1, 2, 0, 1, 1, 0, 1, 2, 1,
                              1, 2, 0, 1, 1, 0, 1, 2, 1,
                              1, 2, 0, 1, 1, 0, 1, 2, 1 };
    const int VALUES_SIZE = sizeof(VALUES)/sizeof(VALUES[0]);
    TimeSeriesDigit ts_in( VALUES, VALUES + VALUES_SIZE);

    RealTime present(gregorian::date(2008,gregorian::Sep,8), posix_time::hours(13) );
    RealDuration delta( posix_time::hours(1) );
    Transformation trans(present, delta);

	TimeSeriesReal ts = create(ts_in, trans );

	BOOST_CHECK_EQUAL(ts.size(), VALUES_SIZE);
	BOOST_CHECK_EQUAL(ts[0].getTime(), present);
	BOOST_CHECK_EQUAL(ts[1].getTime(), present + delta);
	BOOST_CHECK_EQUAL(ts[2].getTime(), present + delta*2);

	//	cout << ts << endl;

}

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



BOOST_AUTO_TEST_SUITE_END()
