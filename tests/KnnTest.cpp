/** plik zawiera test klasyfikatora */
#if defined (_MSC_VER) && (_MSC_VER >= 1400)
//msvc8.0 generuje smieci dla boost::string
#pragma warning(disable:4512)
#pragma warning(disable:4127)
//msvc9.0 warnings for boost::concept_check
#pragma warning(disable:4100)
#endif

#include <iostream>
#include <sstream>
#include <cassert>

#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include <boost/serialization/nvp.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>

#include "../src/learning/Validator.hpp"
#include "../src/learning/KNearestNeighbor.hpp"

using namespace std;
using namespace faif;
using namespace faif::ml;
using boost::unit_test::test_suite;

BOOST_AUTO_TEST_SUITE( FAIF_KNN_classifier_test )

typedef KNearestNeighbor< ValueNominal<string> > KNN;
typedef KNN::AttrIdd AttrIdd;
typedef KNN::AttrDomain AttrDomain;
typedef KNN::Domains Domains;
typedef KNN::Beliefs Beliefs;
typedef KNN::ExampleTest ExampleTest;
typedef KNN::ExampleTrain ExampleTrain;
typedef KNN::ExamplesTrain ExamplesTrain;



BOOST_AUTO_TEST_CASE( classifierMemoryTest ) {

    AttrDomain cat("");
    cat.insert("one"); cat.insert("two");

    AttrDomain a1("a");
    a1.insert("A"); a1.insert("B"); a1.insert("C");
    AttrDomain a2("i");
    a2.insert("I"); a2.insert("J");
    AttrDomain a3("p");
    a3.insert("P");

    Domains d;
    d.push_back(a1); d.push_back(a2); d.push_back(a3);

    KNN knn(d, cat);
    ExamplesTrain ex;
    string ex1[] = {"A","I","P"}; ex.push_back( createExample(ex1, ex1 + 3,"one", knn) );
    string ex2[] = {"A","J","P"}; ex.push_back( createExample(ex2, ex2 + 3,"one", knn) );
    string ex3[] = {"A","I","P"}; ex.push_back( createExample(ex3, ex3 + 3,"one", knn) );
    string ex4[] = {"A","J","P"}; ex.push_back( createExample(ex4, ex4 + 3,"one", knn) );
    string ex5[] = {"C","I","P"}; ex.push_back( createExample(ex5, ex5 + 3,"two", knn) );
    string ex6[] = {"C","J","P"}; ex.push_back( createExample(ex6, ex6 + 3,"two", knn) );
    string ex7[] = {"B","I","P"}; ex.push_back( createExample(ex7, ex7 + 3,"two", knn) );
    string ex8[] = {"B","I","P"}; ex.push_back( createExample(ex8, ex8 + 3,"two", knn) );
    string ex9[] = {"B","I","P"}; ex.push_back( createExample(ex9, ex9 + 3,"two", knn) );

    knn.train(ex);
    // cout << knn << endl;
    stringstream ss;
    ss << knn;

    string s = ss.str().substr(0, ss.str().find('\n')); //first line
    // cout << "XXX" << s << "XXX" << endl;
    BOOST_CHECK( s == string("KNN classifier, defaultK=3, memSize=9:") );

    knn.reset();
    stringstream st;
    st << knn;
    s = st.str().substr(0, st.str().find('\n')); //first line
    BOOST_CHECK( s == string("KNN classifier, defaultK=3, memSize=0:") );
}

BOOST_AUTO_TEST_CASE( classifierClasifyTest ) {

    AttrDomain cat("");
    cat.insert("one"); cat.insert("two"); cat.insert("three");

    AttrDomain a1("a");
    a1.insert("A"); a1.insert("B"); a1.insert("C");
    AttrDomain a2("i");
    a2.insert("I"); a2.insert("J");
    AttrDomain a3("p");
    a3.insert("P");

    Domains d;
    d.push_back(a1); d.push_back(a2); d.push_back(a3);

    KNN knn(d, cat);

    ExamplesTrain ex;
    string ex1[] = {"A","I","P"}; ex.push_back( createExample(ex1, ex1 + 3,"one", knn) );
    string ex2[] = {"A","J","P"}; ex.push_back( createExample(ex2, ex2 + 3,"one", knn) );
    string ex3[] = {"C","I","P"}; ex.push_back( createExample(ex3, ex3 + 3,"three", knn) );
    string ex4[] = {"C","I","P"}; ex.push_back( createExample(ex4, ex4 + 3,"three", knn) );
    string ex5[] = {"C","J","P"}; ex.push_back( createExample(ex5, ex5 + 3,"three", knn) );
    string ex6[] = {"B","I","P"}; ex.push_back( createExample(ex6, ex6 + 3,"two", knn) );
    string ex7[] = {"B","I","P"}; ex.push_back( createExample(ex7, ex7 + 3,"two", knn) );

    string ex8[] = {"B","J","P"}; ExampleTest ex8e = createExample(ex8, ex8 + 3, knn);
    BOOST_CHECK( knn.getCategory(ex8e) == AttrDomain::getUnknownId() ); //empty tree

    knn.train(ex);
    //cout << knn << endl;
    BOOST_CHECK( knn.getCategoryK(ex8e, 4) == knn.getCategoryIdd("two") );
}

Domains createWeatherAttributes() {
    Domains attribs;
    string A1[] = {"sunny", "overcast", "rain" }; attribs.push_back( createDomain("outlook", A1, A1 + 3) );
    string A2[] = {"hot", "mild", "cold"};        attribs.push_back( createDomain("temperature", A2, A2 + 3) );
    string A3[] = {"normal", "high"};             attribs.push_back( createDomain("humidity", A3, A3 + 2) );
    string A4[] = {"strong", "weak"};             attribs.push_back( createDomain("wind", A4, A4 + 2) );
    return attribs;
}

AttrDomain createWeatherCategory() {
    string C[] = {"good","bad"};
    AttrDomain cat = createDomain("", C, C+2);
    return cat;
}

ExamplesTrain createWeatherTrainExamples(const KNN& c) {
    ExamplesTrain ex;
    string E01[] = { "sunny", "hot", "high", "weak"};         ex.push_back( createExample( E01, E01 + 4, "bad", c ) );
    string E02[] = { "sunny", "hot", "high", "strong"};       ex.push_back( createExample( E02, E02 + 4, "bad" , c) );
    string E03[] = { "overcast", "hot", "high", "weak"};      ex.push_back( createExample( E03, E03 + 4, "good" , c) );
    string E04[] = { "rain", "mild", "high", "weak"};         ex.push_back( createExample( E04, E04 + 4, "good" , c) );
    string E05[] = { "rain", "cold", "normal", "weak"};       ex.push_back( createExample( E05, E05 + 4, "good" , c) );
    string E06[] = { "rain", "cold", "normal", "strong"};     ex.push_back( createExample( E06, E06 + 4, "bad" , c) );
    string E07[] = { "overcast", "cold", "normal", "strong"}; ex.push_back( createExample( E07, E07 + 4, "good" , c) );
    string E08[] = { "sunny", "mild", "high", "weak"};        ex.push_back( createExample( E08, E08 + 4, "bad" , c) );
    string E09[] = { "sunny", "cold", "normal", "weak"};      ex.push_back( createExample( E09, E09 + 4, "good" , c) );
    string E10[] = { "rain", "mild", "normal", "weak"};       ex.push_back( createExample( E10, E10 + 4, "good" , c) );
    string E11[] = { "sunny", "mild", "normal", "strong"};    ex.push_back( createExample( E11, E11 + 4, "good" , c) );
    string E12[] = { "overcast", "mild", "high", "strong"};   ex.push_back( createExample( E12, E12 + 4, "good" , c) );
    string E13[] = { "overcast", "hot", "normal", "weak"};    ex.push_back( createExample( E13, E13 + 4, "good" , c) );
    string E14[] = { "rain", "mild", "high", "strong"};       ex.push_back( createExample( E14, E14 + 4, "bad" , c) );
    return ex;
}

BOOST_AUTO_TEST_CASE( weatherClasifierTest ) {

    KNN knn( createWeatherAttributes(), createWeatherCategory() );
    knn.train( createWeatherTrainExamples(knn) );
    knn.setDefaultK(5);
    //cout << knn << endl;
    string ET[] = { "overcast", "hot", "high", "weak"}; ExampleTest et = createExample( ET, ET + 4, knn);
    BOOST_CHECK_EQUAL( knn.getCategory(et), knn.getCategoryIdd("good") );
    string ET01[] = { "sunny", "hot", "high", "weak"};         ExampleTest et01 = createExample( ET01, ET01 + 4, knn);
    BOOST_CHECK( knn.getCategory(et01) == knn.getCategoryIdd("bad") );
    string ET02[] = { "sunny", "hot", "high", "strong"};       ExampleTest et02 = createExample( ET02, ET02 + 4, knn);
    BOOST_CHECK( knn.getCategory(et02) == knn.getCategoryIdd("bad") );
    string ET03[] = { "overcast", "hot", "high", "weak"};      ExampleTest et03 = createExample( ET03, ET03 + 4, knn);
    BOOST_CHECK( knn.getCategory(et03) == knn.getCategoryIdd("good") );
    //knn with k=5 badly classify ET04
    // string ET04[] = { "rain", "mild", "high", "weak"};         ExampleTest et04 = createExample( ET04, ET04 + 4, knn);
    // BOOST_CHECK( knn.getCategory(et04) == knn.getCategoryIdd("good") );
    string ET05[] = { "rain", "cold", "normal", "weak"};       ExampleTest et05 = createExample( ET05, ET05 + 4, knn);
    BOOST_CHECK( knn.getCategory(et05) == knn.getCategoryIdd("good") );
    //knn with k=5 badly classify ET06
    // string ET06[] = { "rain", "cold", "normal", "strong"};     ExampleTest et06 = createExample( ET06, ET06 + 4, knn);
    // BOOST_CHECK( knn.getCategory(et06) == knn.getCategoryIdd("bad") );
    string ET07[] = { "overcast", "cold", "normal", "strong"}; ExampleTest et07 = createExample( ET07, ET07 + 4, knn);
    BOOST_CHECK( knn.getCategory(et07) == knn.getCategoryIdd("good") );
    string ET08[] = { "sunny", "mild", "high", "weak"};        ExampleTest et08 = createExample( ET08, ET08 + 4, knn);
    BOOST_CHECK( knn.getCategory(et08) == knn.getCategoryIdd("bad") );
    string ET09[] = { "sunny", "cold", "normal", "weak"};      ExampleTest et09 = createExample( ET09, ET09 + 4, knn);
    BOOST_CHECK( knn.getCategory(et09) == knn.getCategoryIdd("good") );
    string ET10[] = { "rain", "mild", "normal", "weak"};       ExampleTest et10 = createExample( ET10, ET10 + 4, knn);
    BOOST_CHECK( knn.getCategory(et10) == knn.getCategoryIdd("good") );
    string ET11[] = { "sunny", "mild", "normal", "strong"};    ExampleTest et11 = createExample( ET11, ET11 + 4, knn);
    BOOST_CHECK( knn.getCategory(et11) == knn.getCategoryIdd("good") );
    string ET12[] = { "overcast", "mild", "high", "strong"};   ExampleTest et12 = createExample( ET12, ET12 + 4, knn);
    BOOST_CHECK( knn.getCategory(et12) == knn.getCategoryIdd("good") );
    string ET13[] = { "overcast", "hot", "normal", "weak"};    ExampleTest et13 = createExample( ET13, ET13 + 4, knn);
    BOOST_CHECK( knn.getCategory(et13) == knn.getCategoryIdd("good") );
    string ET14[] = { "rain", "mild", "high", "strong"};       ExampleTest et14 = createExample( ET14, ET14 + 4, knn);
    BOOST_CHECK( knn.getCategory(et14) == knn.getCategoryIdd("bad") );
}

BOOST_AUTO_TEST_CASE( weatherClasifierGetCategories ) {

    KNN knn( createWeatherAttributes(), createWeatherCategory() );
    knn.train( createWeatherTrainExamples(knn) );
    knn.setDefaultK(5);
    //cout << knn << endl;
    ExampleTest et; //empty test example
    Beliefs cwb = knn.getCategories(et); //categories for root
    //cout << cwb << endl;
    BOOST_CHECK_EQUAL( cwb.size(), 2U );
    //the best has index 0
    BOOST_CHECK_EQUAL( cwb[0].getValue(), knn.getCategoryIdd("good") );
    BOOST_CHECK_CLOSE( cwb[0].getProbability(), 0.6, 1.0 );
    BOOST_CHECK_EQUAL( cwb[1].getValue(), knn.getCategoryIdd("bad") );
    BOOST_CHECK_CLOSE( cwb[1].getProbability(), 0.4, 1.0 );

    string ET[] = { "overcast", "hot", "high", "weak"};
    et = createExample( ET, ET + 4, knn);

    cwb = knn.getCategories(et); //categories for root
    BOOST_CHECK( cwb.size() > 0 );
    BOOST_CHECK_EQUAL( cwb.front().getValue(), knn.getCategoryIdd("good") );
    //check the same again
    cwb = knn.getCategories(et); //categories for root
    BOOST_CHECK( cwb.size() > 0 );
    BOOST_CHECK_EQUAL( cwb.front().getValue(), knn.getCategoryIdd("good") );
    //cout << cwb << endl;
}



BOOST_AUTO_TEST_CASE( populationClasifierTest ) {

    /** the survey results; attributes: education level(low, college, acatedmic), age (young, middle, old), gender(female, male)
	 */
    string C[] = {"0","1"};
    AttrDomain cat = createDomain("", C, C+2);
    Domains attribs;
    string A1[] = {"pod", "sre", "wyz" }; attribs.push_back( createDomain("wyk", A1, A1 + 3) );
    string A2[] = {"mlody", "sredn", "stary"};     attribs.push_back( createDomain("wiek", A2, A2 + 3) );
    string A3[] = {"kob", "mez"};          attribs.push_back( createDomain("plec", A3, A3 + 2) );


    KNN knn(attribs, cat);

    //funkcja wynikowa to: c1 <=> (wyk = ww) or (wyk = ws) and (wi = w1) or (wyk = wp) and (wi = w1) and (plec = k)
    ExamplesTrain ex;
    string e01S[] = { "pod", "mlody", "kob" };   ExampleTrain e01 = createExample(e01S, e01S+3, "1" , knn); ex.push_back(e01);
    string e02S[] = { "pod", "mlody", "mez" };   ExampleTrain e02 = createExample(e02S, e02S+3, "0" , knn); ex.push_back(e02);
    string e03S[] = { "pod", "sredn", "kob" };   ExampleTrain e03 = createExample(e03S, e03S+3, "0" , knn); // ex.push_back(e03);
    string e04S[] = { "pod", "sredn", "mez" };   ExampleTrain e04 = createExample(e04S, e04S+3, "0" , knn); ex.push_back(e04);
    string e05S[] = { "pod", "stary", "kob" };   ExampleTrain e05 = createExample(e05S, e05S+3, "0" , knn); ex.push_back(e05);
    string e06S[] = { "pod", "stary", "mez" };   ExampleTrain e06 = createExample(e06S, e06S+3, "0" , knn); ex.push_back(e06);
    string e07S[] = { "sre", "mlody", "kob" };   ExampleTrain e07 = createExample(e07S, e07S+3, "1" , knn); ex.push_back(e07);
    string e08S[] = { "sre", "mlody", "mez" };   ExampleTrain e08 = createExample(e08S, e08S+3, "1" , knn); ex.push_back(e08);
    string e09S[] = { "sre", "sredn", "kob" };   ExampleTrain e09 = createExample(e09S, e09S+3, "0" , knn); ex.push_back(e09);
    string e10S[] = { "sre", "sredn", "mez" };   ExampleTrain e10 = createExample(e10S, e10S+3, "0" , knn); ex.push_back(e10);
    string e11S[] = { "sre", "stary", "kob" };   ExampleTrain e11 = createExample(e11S, e11S+3, "0" , knn); // ex.push_back(e11);
    string e12S[] = { "sre", "stary", "mez" };   ExampleTrain e12 = createExample(e12S, e12S+3, "0" , knn); ex.push_back(e12);
    string e13S[] = { "wyz", "mlody", "kob" };   ExampleTrain e13 = createExample(e13S, e13S+3, "1" , knn); // ex.push_back(e13);
    string e14S[] = { "wyz", "mlody", "mez" };   ExampleTrain e14 = createExample(e14S, e14S+3, "1" , knn); ex.push_back(e14);
    string e15S[] = { "wyz", "sredn", "kob" };   ExampleTrain e15 = createExample(e15S, e15S+3, "1" , knn); // ex.push_back(e15);
    string e16S[] = { "wyz", "sredn", "mez" };   ExampleTrain e16 = createExample(e16S, e16S+3, "1" , knn); ex.push_back(e16);
    string e17S[] = { "wyz", "stary", "kob" };   ExampleTrain e17 = createExample(e17S, e17S+3, "1" , knn); ex.push_back(e17);
    string e18S[] = { "wyz", "stary", "mez" };   ExampleTrain e18 = createExample(e18S, e18S+3, "1" , knn); ex.push_back(e18);

    knn.train(ex);

    //BOOST_CHECK( knn.getCategory(e01) == knn.getCategoryIdd("1") );
    BOOST_CHECK( knn.getCategory(e02) == knn.getCategoryIdd("0") );
    BOOST_CHECK( knn.getCategory(e03) == knn.getCategoryIdd("0") );
    BOOST_CHECK( knn.getCategory(e04) == knn.getCategoryIdd("0") );
    BOOST_CHECK( knn.getCategory(e05) == knn.getCategoryIdd("0") );
    BOOST_CHECK( knn.getCategory(e06) == knn.getCategoryIdd("0") );
    BOOST_CHECK( knn.getCategory(e07) == knn.getCategoryIdd("1") );
    BOOST_CHECK( knn.getCategory(e08) == knn.getCategoryIdd("1") );
    BOOST_CHECK( knn.getCategory(e09) == knn.getCategoryIdd("0") );
    BOOST_CHECK( knn.getCategory(e10) == knn.getCategoryIdd("0") );
    BOOST_CHECK( knn.getCategory(e11) == knn.getCategoryIdd("0") );
    BOOST_CHECK( knn.getCategory(e12) == knn.getCategoryIdd("0") );
    BOOST_CHECK( knn.getCategory(e13) == knn.getCategoryIdd("1") );
    BOOST_CHECK( knn.getCategory(e14) == knn.getCategoryIdd("1") );
    BOOST_CHECK( knn.getCategory(e15) == knn.getCategoryIdd("1") );
    BOOST_CHECK( knn.getCategory(e16) == knn.getCategoryIdd("1") );
    BOOST_CHECK( knn.getCategory(e17) == knn.getCategoryIdd("1") );
    BOOST_CHECK( knn.getCategory(e18) == knn.getCategoryIdd("1") );

    Beliefs cwb = knn.getCategories(e15);
    BOOST_CHECK( cwb.size() > 0U );
    //cout <<  cwb << endl;
    BOOST_CHECK_EQUAL( cwb.front().getValue()->get(), "1");
}

BOOST_AUTO_TEST_CASE( populationClasifierTest2 ) {

    /** the survey results; attributes: education level(low, college, acatedmic), age (young, middle, old), gender(female, male)
	 */
    string C[] = {"0","1", "2"};
    AttrDomain cat = createDomain("", C, C+3);
    Domains attribs;
    string A1[] = {"pod", "sre", "wyz" }; attribs.push_back( createDomain("wyk", A1, A1 + 3) );
    string A2[] = {"mlody", "sredn", "stary"};     attribs.push_back( createDomain("wiek", A2, A2 + 3) );
    string A3[] = {"kob", "mez"};          attribs.push_back( createDomain("plec", A3, A3 + 2) );

    KNN knn(attribs, cat);

    ExamplesTrain ex;
    string e01S[] = { "pod", "mlody", "kob" };   ExampleTrain e01 = createExample(e01S, e01S+3, "1" , knn); ex.push_back(e01);
    string e02S[] = { "pod", "mlody", "mez" };   ExampleTrain e02 = createExample(e02S, e02S+3, "2" , knn); ex.push_back(e02);
    string e03S[] = { "pod", "sredn", "kob" };   ExampleTrain e03 = createExample(e03S, e03S+3, "2" , knn); // ex.push_back(e03);
    string e04S[] = { "pod", "sredn", "mez" };   ExampleTrain e04 = createExample(e04S, e04S+3, "2" , knn); ex.push_back(e04);
    string e05S[] = { "pod", "stary", "kob" };   ExampleTrain e05 = createExample(e05S, e05S+3, "2" , knn); ex.push_back(e05);
    string e06S[] = { "pod", "stary", "mez" };   ExampleTrain e06 = createExample(e06S, e06S+3, "2" , knn); ex.push_back(e06);
    string e07S[] = { "sre", "mlody", "kob" };   ExampleTrain e07 = createExample(e07S, e07S+3, "1" , knn); ex.push_back(e07);
    string e08S[] = { "sre", "mlody", "mez" };   ExampleTrain e08 = createExample(e08S, e08S+3, "1" , knn); ex.push_back(e08);
    string e09S[] = { "sre", "sredn", "kob" };   ExampleTrain e09 = createExample(e09S, e09S+3, "0" , knn); ex.push_back(e09);
    string e10S[] = { "sre", "sredn", "mez" };   ExampleTrain e10 = createExample(e10S, e10S+3, "0" , knn); ex.push_back(e10);
    string e11S[] = { "sre", "stary", "kob" };   ExampleTrain e11 = createExample(e11S, e11S+3, "0" , knn); // ex.push_back(e11);
    string e12S[] = { "sre", "stary", "mez" };   ExampleTrain e12 = createExample(e12S, e12S+3, "0" , knn); ex.push_back(e12);
    string e13S[] = { "wyz", "mlody", "kob" };   ExampleTrain e13 = createExample(e13S, e13S+3, "1" , knn); // ex.push_back(e13);
    string e14S[] = { "wyz", "mlody", "mez" };   ExampleTrain e14 = createExample(e14S, e14S+3, "1" , knn); ex.push_back(e14);
    string e15S[] = { "wyz", "sredn", "kob" };   ExampleTrain e15 = createExample(e15S, e15S+3, "1" , knn); // ex.push_back(e15);
    string e16S[] = { "wyz", "sredn", "mez" };   ExampleTrain e16 = createExample(e16S, e16S+3, "1" , knn); ex.push_back(e16);
    string e17S[] = { "wyz", "stary", "kob" };   ExampleTrain e17 = createExample(e17S, e17S+3, "1" , knn); ex.push_back(e17);
    string e18S[] = { "wyz", "stary", "mez" };   ExampleTrain e18 = createExample(e18S, e18S+3, "1" , knn); ex.push_back(e18);

    knn.train(ex);

    //BOOST_CHECK( knn.getCategory(e01) == knn.getCategoryIdd("1") );
    BOOST_CHECK( knn.getCategory(e02) == knn.getCategoryIdd("2") );
    BOOST_CHECK( knn.getCategory(e03) == knn.getCategoryIdd("2") );
    BOOST_CHECK( knn.getCategory(e04) == knn.getCategoryIdd("2") );
    BOOST_CHECK( knn.getCategory(e05) == knn.getCategoryIdd("2") );
    BOOST_CHECK( knn.getCategory(e06) == knn.getCategoryIdd("2") );
    BOOST_CHECK( knn.getCategory(e07) == knn.getCategoryIdd("1") );
    BOOST_CHECK( knn.getCategory(e08) == knn.getCategoryIdd("1") );
    BOOST_CHECK( knn.getCategory(e09) == knn.getCategoryIdd("0") );
    BOOST_CHECK( knn.getCategory(e10) == knn.getCategoryIdd("0") );
    //BOOST_CHECK( knn.getCategory(e11) == knn.getCategoryIdd("0") );
    BOOST_CHECK( knn.getCategory(e12) == knn.getCategoryIdd("0") );
    BOOST_CHECK( knn.getCategory(e13) == knn.getCategoryIdd("1") );
    BOOST_CHECK( knn.getCategory(e14) == knn.getCategoryIdd("1") );
    BOOST_CHECK( knn.getCategory(e15) == knn.getCategoryIdd("1") );
    BOOST_CHECK( knn.getCategory(e16) == knn.getCategoryIdd("1") );
    BOOST_CHECK( knn.getCategory(e17) == knn.getCategoryIdd("1") );
    BOOST_CHECK( knn.getCategory(e18) == knn.getCategoryIdd("1") );

    Beliefs cwb = knn.getCategories(e15);
    BOOST_CHECK( cwb.size() > 0U );
    BOOST_CHECK_EQUAL( cwb[0].getValue()->get(), "1");
}

BOOST_AUTO_TEST_CASE( internalStateTest ) {

    KNN knn( createWeatherAttributes(), createWeatherCategory() );
	knn.setDefaultK(5);
    knn.train( createWeatherTrainExamples(knn) );

    ostringstream oss;
    boost::archive::xml_oarchive oa(oss);

    oa << boost::serialization::make_nvp("KNN", knn );

    string dtcSerialString = oss.str();
    //cout << oss.str() << endl;

    {
        //check if restoring object from given string and serializing this restored object gives the string
        //of equal length
        std::istringstream iss(dtcSerialString);
        boost::archive::xml_iarchive ia(iss);

        KNN nnn;
        ia >> boost::serialization::make_nvp("KNN", nnn);
        //cout << nnn;
        ostringstream oss2;
        boost::archive::xml_oarchive oa2(oss2);

        oa2 << boost::serialization::make_nvp("KNN", nnn );

        BOOST_CHECK_EQUAL(dtcSerialString.size(), oss2.str().size() );
    }

    //setting
    KNN empty;

    std::istringstream iss(dtcSerialString);
    boost::archive::xml_iarchive ia(iss);
    ia >> boost::serialization::make_nvp("KNN", empty);
    //cout << empty << endl;

    string ET[] = { "overcast", "hot", "high", "weak"}; ExampleTest et = createExample( ET, ET + 4, empty);
    BOOST_CHECK_EQUAL( empty.getCategory(et), empty.getCategoryIdd("good") );

    string ET01[] = { "sunny", "hot", "high", "weak"};         ExampleTest et01 = createExample( ET01, ET01 + 4, empty);
    BOOST_CHECK( empty.getCategory(et01) == empty.getCategoryIdd("bad") );
    string ET02[] = { "sunny", "hot", "high", "strong"};       ExampleTest et02 = createExample( ET02, ET02 + 4, empty);
    BOOST_CHECK( empty.getCategory(et02) == empty.getCategoryIdd("bad") );
    string ET03[] = { "overcast", "hot", "high", "weak"};      ExampleTest et03 = createExample( ET03, ET03 + 4, empty);
    BOOST_CHECK( empty.getCategory(et03) == empty.getCategoryIdd("good") );
    // string ET04[] = { "rain", "mild", "high", "weak"};         ExampleTest et04 = createExample( ET04, ET04 + 4, empty);
    // BOOST_CHECK( empty.getCategory(et04) == empty.getCategoryIdd("good") );
    string ET05[] = { "rain", "cold", "normal", "weak"};       ExampleTest et05 = createExample( ET05, ET05 + 4, empty);
    BOOST_CHECK( empty.getCategory(et05) == empty.getCategoryIdd("good") );
    // string ET06[] = { "rain", "cold", "normal", "strong"};     ExampleTest et06 = createExample( ET06, ET06 + 4, empty);
    // BOOST_CHECK( empty.getCategory(et06) == empty.getCategoryIdd("bad") );
    string ET07[] = { "overcast", "cold", "normal", "strong"}; ExampleTest et07 = createExample( ET07, ET07 + 4, empty);
    BOOST_CHECK( empty.getCategory(et07) == empty.getCategoryIdd("good") );
    string ET08[] = { "sunny", "mild", "high", "weak"};        ExampleTest et08 = createExample( ET08, ET08 + 4, empty);
    BOOST_CHECK( empty.getCategory(et08) == empty.getCategoryIdd("bad") );
    string ET09[] = { "sunny", "cold", "normal", "weak"};      ExampleTest et09 = createExample( ET09, ET09 + 4, empty);
    BOOST_CHECK( empty.getCategory(et09) == empty.getCategoryIdd("good") );
    string ET10[] = { "rain", "mild", "normal", "weak"};       ExampleTest et10 = createExample( ET10, ET10 + 4, empty);
    BOOST_CHECK( empty.getCategory(et10) == empty.getCategoryIdd("good") );
    string ET11[] = { "sunny", "mild", "normal", "strong"};    ExampleTest et11 = createExample( ET11, ET11 + 4, empty);
    BOOST_CHECK( empty.getCategory(et11) == empty.getCategoryIdd("good") );
    string ET12[] = { "overcast", "mild", "high", "strong"};   ExampleTest et12 = createExample( ET12, ET12 + 4, empty);
    BOOST_CHECK( empty.getCategory(et12) == empty.getCategoryIdd("good") );
    string ET13[] = { "overcast", "hot", "normal", "weak"};    ExampleTest et13 = createExample( ET13, ET13 + 4, empty);
    BOOST_CHECK( empty.getCategory(et13) == empty.getCategoryIdd("good") );
    string ET14[] = { "rain", "mild", "high", "strong"};       ExampleTest et14 = createExample( ET14, ET14 + 4, empty);
    BOOST_CHECK( empty.getCategory(et14) == empty.getCategoryIdd("bad") );
}

BOOST_AUTO_TEST_CASE( knnValidator ) {
    KNN knn( createWeatherAttributes(), createWeatherCategory() );
    knn.setDefaultK(5);
    ExamplesTrain ex = createWeatherTrainExamples(knn);
    knn.train(ex);

    BOOST_CHECK( checkClassifier(ex, knn) >= static_cast<int>(ex.size()) - 2 );
    BOOST_CHECK_MESSAGE( checkCross(ex, 14, knn) >= 0.4, "Random cross-validator test, it may produce false alarms");
}



BOOST_AUTO_TEST_SUITE_END()
