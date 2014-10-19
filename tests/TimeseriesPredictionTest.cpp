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
#include "../src/timeseries/TimeseriesExceptions.hpp"
#include "../src/timeseries/Predictions.hpp"

using namespace std;
using namespace boost;
using namespace faif::timeseries;
using boost::unit_test::test_suite;

BOOST_AUTO_TEST_SUITE( faif_timeseries_prediction_test )

BOOST_AUTO_TEST_CASE( ARTestGeom )
{
    ARDef ardef(1); ardef[0] = 2;

    const double PAST_VALUES[] = { 0.1 };
    const int PAST_VALUES_SIZE = sizeof(PAST_VALUES)/sizeof(PAST_VALUES[0]);
    TimeSeriesDigit past( PAST_VALUES, PAST_VALUES + PAST_VALUES_SIZE);

    PredictionAR ar( TimeSeriesDigit(past, -PAST_VALUES_SIZE), ardef);
    //cout << "history: " << ar.getHistory() << endl;
    //cout << TimeSeriesDigit(past, -PAST_VALUES_SIZE) << endl;

    const double PRED_VALUES[] = { 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8 };
    const int PRED_VALUES_SIZE = sizeof(PRED_VALUES)/sizeof(PRED_VALUES[0]);
    TimeSeriesDigit pred( PRED_VALUES, PRED_VALUES + PRED_VALUES_SIZE);

    TimeSeriesDigit calc = ar.calculatePrediction(0, pred.size()-1 );
    BOOST_CHECK_EQUAL_COLLECTIONS( calc.begin(), calc.end(), pred.begin(), pred.end() );
}

BOOST_AUTO_TEST_CASE( ARTestFibb )
{
    ARDef ardef(2); ardef[0]=1; ardef[1]=1;

    const double PAST_VALUES[] = { 1, 1 };
    const int PAST_VALUES_SIZE = sizeof(PAST_VALUES)/sizeof(PAST_VALUES[0]);
    TimeSeriesDigit past( PAST_VALUES, PAST_VALUES + PAST_VALUES_SIZE);

    PredictionAR ar( TimeSeriesDigit(past, -PAST_VALUES_SIZE), ardef);

    const double PRED_VALUES[] = { 2, 3, 5, 8, 13, 21, 34 };
    const int PRED_VALUES_SIZE = sizeof(PRED_VALUES)/sizeof(PRED_VALUES[0]);
    TimeSeriesDigit pred( PRED_VALUES, PRED_VALUES + PRED_VALUES_SIZE);

    TimeSeriesDigit calc = ar.calculatePrediction(0, pred.size() - 1);
    BOOST_CHECK_EQUAL_COLLECTIONS( calc.begin(), calc.end(), pred.begin(), pred.end() );
}

BOOST_AUTO_TEST_CASE( ARTestBipol ) {
    ARDef ardef(1); ardef[0]=-1;

    const double PAST_VALUES[] = { 1 };
    const int PAST_VALUES_SIZE = sizeof(PAST_VALUES)/sizeof(PAST_VALUES[0]);
    TimeSeriesDigit past( PAST_VALUES, PAST_VALUES + PAST_VALUES_SIZE);

    PredictionAR ar( TimeSeriesDigit(past, -PAST_VALUES_SIZE), ardef);

    const double PRED_VALUES[] = { -1, 1, -1, 1, -1, 1 };
    const int PRED_VALUES_SIZE = sizeof(PRED_VALUES)/sizeof(PRED_VALUES[0]);
    TimeSeriesDigit pred( PRED_VALUES, PRED_VALUES + PRED_VALUES_SIZE);

    TimeSeriesDigit calc = ar.calculatePrediction(0, pred.size() - 1);
    BOOST_CHECK_EQUAL_COLLECTIONS( calc.begin(), calc.end(), pred.begin(), pred.end() );
}

BOOST_AUTO_TEST_CASE( CheckARWithPast) {
    ARDef ardef(1);
    ardef[0]=-1;

    const double PAST_VALUES[] = { 0.1, 0.2, 0.5, 1 };
    const int PAST_VALUES_SIZE = sizeof(PAST_VALUES)/sizeof(PAST_VALUES[0]);
    TimeSeriesDigit past( PAST_VALUES, PAST_VALUES + PAST_VALUES_SIZE);

    PredictionAR ar( TimeSeriesDigit(past, -PAST_VALUES_SIZE), ardef);

    const double PRED_VALUES[] = { 0.1, 0.2, 0.5, 1, -1, 1, -1, 1, -1, 1 };
    const int PRED_VALUES_SIZE = sizeof(PRED_VALUES)/sizeof(PRED_VALUES[0]);
    TimeSeriesDigit future( TimeSeriesDigit( PRED_VALUES, PRED_VALUES + PRED_VALUES_SIZE), -PAST_VALUES_SIZE );

    TimeSeriesDigit calc = ar.calculatePrediction(-PAST_VALUES_SIZE, PRED_VALUES_SIZE - PAST_VALUES_SIZE - 1);
    BOOST_CHECK_EQUAL_COLLECTIONS(future.begin(), future.end(), calc.begin(), calc.end() );
}

namespace {

    typedef TimeSeriesDigit::const_iterator TSIter;
    void checkCloseCollection( TSIter begin1, TSIter end1, TSIter begin2, TSIter end2 ) {
        for(;begin1 != end1 && begin2 != end2;++begin1,++begin2) {
            BOOST_CHECK_CLOSE( begin1->getValue(), begin2->getValue(), 0.1 );
            BOOST_CHECK_EQUAL( begin1->getTime(), begin2->getTime() );
        }
        BOOST_CHECK_MESSAGE( begin1 == end1, "collection A shorter than collection B" );
        BOOST_CHECK_MESSAGE( begin2 == end2, "collection B shorter than collection A" );
    }

} //namespace


BOOST_AUTO_TEST_CASE( KNNTest ) {
    const double PAST_VALUES[] = { 0.1, 0.1, -0.1, -0.1, 0.1, 0.1, -0.1, -0.1,
                                   0.1, 0.1, -0.1, -0.1, 0.1, 0.1, -0.1, -0.1,
                                   0.1, 0.1, -0.1, -0.1, 0.1, 0.1, -0.1, -0.1,
                                   0.1, 0.1, -0.1, -0.1, 0.1, 0.1, -0.1, -0.1 };
    const int PAST_VALUES_SIZE = sizeof(PAST_VALUES)/sizeof(PAST_VALUES[0]);
    TimeSeriesDigit ts_in( PAST_VALUES, PAST_VALUES + PAST_VALUES_SIZE);

    //KNN - jeden sasiad , blok od TimeDigit = -4 do -1 (cztery probki sa badane)
    PredictionKNN knn( TimeSeriesDigit(ts_in, -PAST_VALUES_SIZE), KNNDef(1, 4) );

    TimeSeriesDigit ts_out = knn.calculatePrediction(0,3);

    checkCloseCollection( ts_out.begin(), ts_out.end(), ts_in.begin(), ts_in.begin() + ts_out.size() );

    PredictionKNN knn2( TimeSeriesDigit(ts_in, -PAST_VALUES_SIZE), KNNDef(3, 4) );
    TimeSeriesDigit ts_out2 = knn2.calculatePrediction(0, 3);
    checkCloseCollection( ts_out.begin(), ts_out.end(), ts_out2.begin(), ts_out2.end() );
}


BOOST_AUTO_TEST_CASE( KNN_short_input_test ) {
    const double PAST_VALUES[] = { 0.1 };
    const int PAST_VALUES_SIZE = sizeof(PAST_VALUES)/sizeof(PAST_VALUES[0]);
    TimeSeriesDigit ts_in( PAST_VALUES, PAST_VALUES + PAST_VALUES_SIZE);

    //KNN - three neighbours , block from TimeDigit = -4 to TimeDigit = -1 (four values length)
    PredictionKNN knn( TimeSeriesDigit(ts_in, -PAST_VALUES_SIZE), KNNDef(3, 4) );
    BOOST_CHECK_NO_THROW( knn.calculatePrediction(0,20) );
}

BOOST_AUTO_TEST_CASE( KNNLongPredictionTest ) {
    const double PAST_VALUES[] = { 0.1, 0.1, -0.1, -0.1, 0.1, 0.1, -0.1, -0.1,
                                   0.1, 0.1, -0.1, -0.1, 0.1, 0.1, -0.1, -0.1,
                                   0.1, 0.1, -0.1, -0.1, 0.1, 0.1, -0.1, -0.1,
                                   0.1, 0.1, -0.1, -0.1, 0.1, 0.1, -0.1, -0.1 };
    const int PAST_VALUES_SIZE = sizeof(PAST_VALUES)/sizeof(PAST_VALUES[0]);
    TimeSeriesDigit ts_in( PAST_VALUES, PAST_VALUES + PAST_VALUES_SIZE);

    //KNN - jeden sasiad , blok od TimeDigit = -4 do -1 (cztery probki sa badane)
    PredictionKNN knn( TimeSeriesDigit(ts_in, -PAST_VALUES_SIZE), KNNDef(1, 4) );

    TimeSeriesDigit ts_out = knn.calculatePrediction(0, 100);

    BOOST_CHECK_EQUAL( ts_out.size(), 101U );

    BOOST_CHECK_CLOSE( ts_out[80].getValue(), 0.0, 0.5 );
}

BOOST_AUTO_TEST_CASE( KNNPastPredictionTest ) {
	//TODO
	// dla danych (DigitTimeseries?), ktore koncza sie daleko w przeszlosci, np. -1000, -999, -998, ..., -900
	//jezeli chcemy obliczac predykcje KNN to blok referencyjny jest brany jako N ostatnich dostepnych probek,
	//tzn. dla N=4 (-903, -902, -901, -900), natomiast predykcja jest dla t = 0, 1, 2 ,... , X
	//wiec pomijane sa nieistniejace dane (nie sa brane pod uwage).

	// Innymi slowy zaklada, ze ts ma probki dla t = -X, ..., -4, -3, -2, -1

	// Poprawa:
	// blok referencyjny: ostatnie dostepne probki w ts, ale obliczany jest offset,
	// jezeli ostatnia dostepna to t = -900 to offset = 899
	// nastepnie znajdowany nalepszy blok w historii (powiedzmy dlugosc bloku referencyjnego R=4,
	// najlepsze to jest to t_best = (-1000, -999, -998, -997)
	// * predykcja sa to probki dla t_best + offset + R



    const double PAST_VALUES[] = { 269, 275, 220, 275, 261, 276, 318, 268, 318, 317,
                                   312, 276, 317, 324, 345, 290, 331, 331, 326, 319,
                                   333, 318, 306, 304, 320, 297, 318, 352, 622, 712 };

    const int PAST_VALUES_SIZE = sizeof(PAST_VALUES)/sizeof(PAST_VALUES[0]);
    TimeSeriesDigit ts_past( PAST_VALUES, PAST_VALUES + PAST_VALUES_SIZE);
    TimeSeriesDigit ts_in(ts_past, -130000);

    //cout << ts_in << endl;

    //KNN - jeden sasiad , blok od TimeDigit = -4 do -1 (cztery probki sa badane)
    PredictionKNN knn( ts_in, KNNDef(1, 24) );

    TimeSeriesDigit ts_out = knn.calculatePrediction(0, 24);
    //cout << ts_out << endl;
}


BOOST_AUTO_TEST_SUITE_END()
