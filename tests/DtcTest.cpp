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
#include "../src/learning/DecisionTree.hpp"

using namespace std;
using namespace faif;
using namespace faif::ml;
using boost::unit_test::test_suite;

BOOST_AUTO_TEST_SUITE( FAIF_DTC_classifier_test )

typedef DecisionTree< ValueNominal<string> > DTC;
typedef DTC::AttrIdd AttrIdd;
typedef DTC::AttrDomain AttrDomain;
typedef DTC::Domains Domains;
typedef DTC::Beliefs Beliefs;
typedef DTC::ExampleTest ExampleTest;
typedef DTC::ExampleTrain ExampleTrain;
typedef DTC::ExamplesTrain ExamplesTrain;

BOOST_AUTO_TEST_CASE( classifierCounterTest ) {

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

    DTC dt(d, cat);
    ExamplesTrain ex;
    string ex1[] = {"A","I","P"}; ex.push_back( createExample(ex1, ex1 + 3,"one", dt) );
    string ex2[] = {"A","J","P"}; ex.push_back( createExample(ex2, ex2 + 3,"one", dt) );
    string ex3[] = {"A","I","P"}; ex.push_back( createExample(ex3, ex3 + 3,"one", dt) );
    string ex4[] = {"A","J","P"}; ex.push_back( createExample(ex4, ex4 + 3,"one", dt) );
    string ex5[] = {"C","I","P"}; ex.push_back( createExample(ex5, ex5 + 3,"two", dt) );
    string ex6[] = {"C","J","P"}; ex.push_back( createExample(ex6, ex6 + 3,"two", dt) );
    string ex7[] = {"B","I","P"}; ex.push_back( createExample(ex7, ex7 + 3,"two", dt) );
    string ex8[] = {"B","I","P"}; ex.push_back( createExample(ex8, ex8 + 3,"two", dt) );
    string ex9[] = {"B","I","P"}; ex.push_back( createExample(ex9, ex9 + 3,"two", dt) );

    dt.train(ex);
    const char* OUTPUT_STRING =
        "Internal (Major:two, test:Domain: a, Value:A);\n"
        " Leaf (Major:one);\n"
        " Leaf (Major:two);\n";
    stringstream ss;
    ss << dt;
    string s = ss.str();
    // cout << "XXX" << s << "XXX" << endl;
    // cout << "YYY" << string(OUTPUT_STRING) << "YYY" << endl;
    BOOST_CHECK( s == string(OUTPUT_STRING) );

    dt.reset();
    stringstream st;
    st << dt;
    s = st.str();
    BOOST_CHECK( s == string("Empty DTC\n") );
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

    DTC dt(d, cat);

    ExamplesTrain ex;
    string ex1[] = {"A","I","P"}; ex.push_back( createExample(ex1, ex1 + 3,"one", dt) );
    string ex2[] = {"A","J","P"}; ex.push_back( createExample(ex2, ex2 + 3,"one", dt) );
    string ex3[] = {"C","I","P"}; ex.push_back( createExample(ex3, ex3 + 3,"three", dt) );
    string ex4[] = {"C","I","P"}; ex.push_back( createExample(ex4, ex4 + 3,"three", dt) );
    string ex5[] = {"C","J","P"}; ex.push_back( createExample(ex5, ex5 + 3,"three", dt) );
    string ex6[] = {"B","I","P"}; ex.push_back( createExample(ex6, ex6 + 3,"two", dt) );
    string ex7[] = {"B","I","P"}; ex.push_back( createExample(ex7, ex7 + 3,"two", dt) );

    string ex8[] = {"B","J","P"}; ExampleTest ex8e = createExample(ex8, ex8 + 3, dt);
    BOOST_CHECK( dt.getCategory(ex8e) == AttrDomain::getUnknownId() ); //empty tree

    dt.train(ex);
    //cout << dt << endl;
    BOOST_CHECK( dt.getCategory(ex8e) == dt.getCategoryIdd("two") );
    ExampleTest ex9; //empty example
    BOOST_CHECK( dt.getCategory(ex9) == dt.getCategoryIdd("three") ); //major category at root
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

ExamplesTrain createWeatherTrainExamples(const DTC& c) {
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

    DTC dt( createWeatherAttributes(), createWeatherCategory() );
    DecisionTreeTrainParam p = dt.getTrainParam();
    p.allowedNbrMiscEx = 0;
    dt.setTrainParam(p);
    dt.train( createWeatherTrainExamples(dt) );
    //cout << dt << endl;
    string ET[] = { "overcast", "hot", "high", "weak"}; ExampleTest et = createExample( ET, ET + 4, dt);
    BOOST_CHECK_EQUAL( dt.getCategory(et), dt.getCategoryIdd("good") );

    string ET01[] = { "sunny", "hot", "high", "weak"};         ExampleTest et01 = createExample( ET01, ET01 + 4, dt);
    BOOST_CHECK( dt.getCategory(et01) == dt.getCategoryIdd("bad") );
    string ET02[] = { "sunny", "hot", "high", "strong"};       ExampleTest et02 = createExample( ET02, ET02 + 4, dt);
    BOOST_CHECK( dt.getCategory(et02) == dt.getCategoryIdd("bad") );
    string ET03[] = { "overcast", "hot", "high", "weak"};      ExampleTest et03 = createExample( ET03, ET03 + 4, dt);
    BOOST_CHECK( dt.getCategory(et03) == dt.getCategoryIdd("good") );
    string ET04[] = { "rain", "mild", "high", "weak"};         ExampleTest et04 = createExample( ET04, ET04 + 4, dt);
    BOOST_CHECK( dt.getCategory(et04) == dt.getCategoryIdd("good") );
    string ET05[] = { "rain", "cold", "normal", "weak"};       ExampleTest et05 = createExample( ET05, ET05 + 4, dt);
    BOOST_CHECK( dt.getCategory(et05) == dt.getCategoryIdd("good") );
    string ET06[] = { "rain", "cold", "normal", "strong"};     ExampleTest et06 = createExample( ET06, ET06 + 4, dt);
    BOOST_CHECK( dt.getCategory(et06) == dt.getCategoryIdd("bad") );
    string ET07[] = { "overcast", "cold", "normal", "strong"}; ExampleTest et07 = createExample( ET07, ET07 + 4, dt);
    BOOST_CHECK( dt.getCategory(et07) == dt.getCategoryIdd("good") );
    string ET08[] = { "sunny", "mild", "high", "weak"};        ExampleTest et08 = createExample( ET08, ET08 + 4, dt);
    BOOST_CHECK( dt.getCategory(et08) == dt.getCategoryIdd("bad") );
    string ET09[] = { "sunny", "cold", "normal", "weak"};      ExampleTest et09 = createExample( ET09, ET09 + 4, dt);
    BOOST_CHECK( dt.getCategory(et09) == dt.getCategoryIdd("good") );
    string ET10[] = { "rain", "mild", "normal", "weak"};       ExampleTest et10 = createExample( ET10, ET10 + 4, dt);
    BOOST_CHECK( dt.getCategory(et10) == dt.getCategoryIdd("good") );
    string ET11[] = { "sunny", "mild", "normal", "strong"};    ExampleTest et11 = createExample( ET11, ET11 + 4, dt);
    BOOST_CHECK( dt.getCategory(et11) == dt.getCategoryIdd("good") );
    string ET12[] = { "overcast", "mild", "high", "strong"};   ExampleTest et12 = createExample( ET12, ET12 + 4, dt);
    BOOST_CHECK( dt.getCategory(et12) == dt.getCategoryIdd("good") );
    string ET13[] = { "overcast", "hot", "normal", "weak"};    ExampleTest et13 = createExample( ET13, ET13 + 4, dt);
    BOOST_CHECK( dt.getCategory(et13) == dt.getCategoryIdd("good") );
    string ET14[] = { "rain", "mild", "high", "strong"};       ExampleTest et14 = createExample( ET14, ET14 + 4, dt);
    BOOST_CHECK( dt.getCategory(et14) == dt.getCategoryIdd("bad") );
}

BOOST_AUTO_TEST_CASE( weatherClasifierGetCategories ) {

    DTC dt( createWeatherAttributes(), createWeatherCategory() );
    dt.train( createWeatherTrainExamples(dt) );
    //cout << dt << endl;
    ExampleTest et; //empty test example
    Beliefs cwb = dt.getCategories(et); //categories for root
    //cout << cwb << endl;
    BOOST_CHECK_EQUAL( cwb.size(), 2U );
    //the best has index 0
    BOOST_CHECK_EQUAL( cwb[0].getValue()->get(), "good");
    BOOST_CHECK_CLOSE( cwb[0].getProbability(), 0.643, 1.0 );
    BOOST_CHECK_EQUAL( cwb[1].getValue()->get(), "bad");
    BOOST_CHECK_CLOSE( cwb[1].getProbability(), 0.357, 1.0 );

    string ET[] = { "overcast", "hot", "high", "weak"};
    et = createExample( ET, ET + 4, dt);

    cwb = dt.getCategories(et); //categories for root
    BOOST_CHECK( cwb.size() > 0 );
    BOOST_CHECK_EQUAL( cwb.front().getValue(), dt.getCategoryIdd("good") );
    //check the same again
    cwb = dt.getCategories(et); //categories for root
    BOOST_CHECK( cwb.size() > 0 );
    BOOST_CHECK_EQUAL( cwb.front().getValue(), dt.getCategoryIdd("good") );
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


    DTC dt(attribs, cat);

    //funkcja wynikowa to: c1 <=> (wyk = ww) or (wyk = ws) and (wi = w1) or (wyk = wp) and (wi = w1) and (plec = k)
    ExamplesTrain ex;
    string e01S[] = { "pod", "mlody", "kob" };   ExampleTrain e01 = createExample(e01S, e01S+3, "1" , dt); ex.push_back(e01);
    string e02S[] = { "pod", "mlody", "mez" };   ExampleTrain e02 = createExample(e02S, e02S+3, "0" , dt); ex.push_back(e02);
    string e03S[] = { "pod", "sredn", "kob" };   ExampleTrain e03 = createExample(e03S, e03S+3, "0" , dt); // ex.push_back(e03);
    string e04S[] = { "pod", "sredn", "mez" };   ExampleTrain e04 = createExample(e04S, e04S+3, "0" , dt); ex.push_back(e04);
    string e05S[] = { "pod", "stary", "kob" };   ExampleTrain e05 = createExample(e05S, e05S+3, "0" , dt); ex.push_back(e05);
    string e06S[] = { "pod", "stary", "mez" };   ExampleTrain e06 = createExample(e06S, e06S+3, "0" , dt); ex.push_back(e06);
    string e07S[] = { "sre", "mlody", "kob" };   ExampleTrain e07 = createExample(e07S, e07S+3, "1" , dt); ex.push_back(e07);
    string e08S[] = { "sre", "mlody", "mez" };   ExampleTrain e08 = createExample(e08S, e08S+3, "1" , dt); ex.push_back(e08);
    string e09S[] = { "sre", "sredn", "kob" };   ExampleTrain e09 = createExample(e09S, e09S+3, "0" , dt); ex.push_back(e09);
    string e10S[] = { "sre", "sredn", "mez" };   ExampleTrain e10 = createExample(e10S, e10S+3, "0" , dt); ex.push_back(e10);
    string e11S[] = { "sre", "stary", "kob" };   ExampleTrain e11 = createExample(e11S, e11S+3, "0" , dt); // ex.push_back(e11);
    string e12S[] = { "sre", "stary", "mez" };   ExampleTrain e12 = createExample(e12S, e12S+3, "0" , dt); ex.push_back(e12);
    string e13S[] = { "wyz", "mlody", "kob" };   ExampleTrain e13 = createExample(e13S, e13S+3, "1" , dt); // ex.push_back(e13);
    string e14S[] = { "wyz", "mlody", "mez" };   ExampleTrain e14 = createExample(e14S, e14S+3, "1" , dt); ex.push_back(e14);
    string e15S[] = { "wyz", "sredn", "kob" };   ExampleTrain e15 = createExample(e15S, e15S+3, "1" , dt); // ex.push_back(e15);
    string e16S[] = { "wyz", "sredn", "mez" };   ExampleTrain e16 = createExample(e16S, e16S+3, "1" , dt); ex.push_back(e16);
    string e17S[] = { "wyz", "stary", "kob" };   ExampleTrain e17 = createExample(e17S, e17S+3, "1" , dt); ex.push_back(e17);
    string e18S[] = { "wyz", "stary", "mez" };   ExampleTrain e18 = createExample(e18S, e18S+3, "1" , dt); ex.push_back(e18);

    dt.train(ex);

    BOOST_CHECK( dt.getCategory(e01) == dt.getCategoryIdd("1") );
    //BOOST_CHECK( dt.getCategory(e02) == dt.getCategoryIdd("0") );
    BOOST_CHECK( dt.getCategory(e03) == dt.getCategoryIdd("0") );
    BOOST_CHECK( dt.getCategory(e04) == dt.getCategoryIdd("0") );
    BOOST_CHECK( dt.getCategory(e05) == dt.getCategoryIdd("0") );
    BOOST_CHECK( dt.getCategory(e06) == dt.getCategoryIdd("0") );
    BOOST_CHECK( dt.getCategory(e07) == dt.getCategoryIdd("1") );
    BOOST_CHECK( dt.getCategory(e08) == dt.getCategoryIdd("1") );
    BOOST_CHECK( dt.getCategory(e09) == dt.getCategoryIdd("0") );
    BOOST_CHECK( dt.getCategory(e10) == dt.getCategoryIdd("0") );
    BOOST_CHECK( dt.getCategory(e11) == dt.getCategoryIdd("0") );
    BOOST_CHECK( dt.getCategory(e12) == dt.getCategoryIdd("0") );
    BOOST_CHECK( dt.getCategory(e13) == dt.getCategoryIdd("1") );
    BOOST_CHECK( dt.getCategory(e14) == dt.getCategoryIdd("1") );
    BOOST_CHECK( dt.getCategory(e15) == dt.getCategoryIdd("1") );
    BOOST_CHECK( dt.getCategory(e16) == dt.getCategoryIdd("1") );
    BOOST_CHECK( dt.getCategory(e17) == dt.getCategoryIdd("1") );
    BOOST_CHECK( dt.getCategory(e18) == dt.getCategoryIdd("1") );

    Beliefs cwb = dt.getCategories(e15);
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

    DTC dt(attribs, cat);

    ExamplesTrain ex;
    string e01S[] = { "pod", "mlody", "kob" };   ExampleTrain e01 = createExample(e01S, e01S+3, "1" , dt); ex.push_back(e01);
    string e02S[] = { "pod", "mlody", "mez" };   ExampleTrain e02 = createExample(e02S, e02S+3, "2" , dt); ex.push_back(e02);
    string e03S[] = { "pod", "sredn", "kob" };   ExampleTrain e03 = createExample(e03S, e03S+3, "2" , dt); // ex.push_back(e03);
    string e04S[] = { "pod", "sredn", "mez" };   ExampleTrain e04 = createExample(e04S, e04S+3, "2" , dt); ex.push_back(e04);
    string e05S[] = { "pod", "stary", "kob" };   ExampleTrain e05 = createExample(e05S, e05S+3, "2" , dt); ex.push_back(e05);
    string e06S[] = { "pod", "stary", "mez" };   ExampleTrain e06 = createExample(e06S, e06S+3, "2" , dt); ex.push_back(e06);
    string e07S[] = { "sre", "mlody", "kob" };   ExampleTrain e07 = createExample(e07S, e07S+3, "1" , dt); ex.push_back(e07);
    string e08S[] = { "sre", "mlody", "mez" };   ExampleTrain e08 = createExample(e08S, e08S+3, "1" , dt); ex.push_back(e08);
    string e09S[] = { "sre", "sredn", "kob" };   ExampleTrain e09 = createExample(e09S, e09S+3, "0" , dt); ex.push_back(e09);
    string e10S[] = { "sre", "sredn", "mez" };   ExampleTrain e10 = createExample(e10S, e10S+3, "0" , dt); ex.push_back(e10);
    string e11S[] = { "sre", "stary", "kob" };   ExampleTrain e11 = createExample(e11S, e11S+3, "0" , dt); // ex.push_back(e11);
    string e12S[] = { "sre", "stary", "mez" };   ExampleTrain e12 = createExample(e12S, e12S+3, "0" , dt); ex.push_back(e12);
    string e13S[] = { "wyz", "mlody", "kob" };   ExampleTrain e13 = createExample(e13S, e13S+3, "1" , dt); // ex.push_back(e13);
    string e14S[] = { "wyz", "mlody", "mez" };   ExampleTrain e14 = createExample(e14S, e14S+3, "1" , dt); ex.push_back(e14);
    string e15S[] = { "wyz", "sredn", "kob" };   ExampleTrain e15 = createExample(e15S, e15S+3, "1" , dt); // ex.push_back(e15);
    string e16S[] = { "wyz", "sredn", "mez" };   ExampleTrain e16 = createExample(e16S, e16S+3, "1" , dt); ex.push_back(e16);
    string e17S[] = { "wyz", "stary", "kob" };   ExampleTrain e17 = createExample(e17S, e17S+3, "1" , dt); ex.push_back(e17);
    string e18S[] = { "wyz", "stary", "mez" };   ExampleTrain e18 = createExample(e18S, e18S+3, "1" , dt); ex.push_back(e18);

    dt.train(ex);

    //BOOST_CHECK( dt.getCategory(e01) == dt.getCategoryIdd("1") );
    BOOST_CHECK( dt.getCategory(e02) == dt.getCategoryIdd("2") );
    BOOST_CHECK( dt.getCategory(e03) == dt.getCategoryIdd("2") );
    BOOST_CHECK( dt.getCategory(e04) == dt.getCategoryIdd("2") );
    BOOST_CHECK( dt.getCategory(e05) == dt.getCategoryIdd("2") );
    BOOST_CHECK( dt.getCategory(e06) == dt.getCategoryIdd("2") );
    BOOST_CHECK( dt.getCategory(e07) == dt.getCategoryIdd("1") );
    BOOST_CHECK( dt.getCategory(e08) == dt.getCategoryIdd("1") );
    BOOST_CHECK( dt.getCategory(e09) == dt.getCategoryIdd("0") );
    BOOST_CHECK( dt.getCategory(e10) == dt.getCategoryIdd("0") );
    BOOST_CHECK( dt.getCategory(e11) == dt.getCategoryIdd("0") );
    BOOST_CHECK( dt.getCategory(e12) == dt.getCategoryIdd("0") );
    BOOST_CHECK( dt.getCategory(e13) == dt.getCategoryIdd("1") );
    BOOST_CHECK( dt.getCategory(e14) == dt.getCategoryIdd("1") );
    BOOST_CHECK( dt.getCategory(e15) == dt.getCategoryIdd("1") );
    BOOST_CHECK( dt.getCategory(e16) == dt.getCategoryIdd("1") );
    BOOST_CHECK( dt.getCategory(e17) == dt.getCategoryIdd("1") );
    BOOST_CHECK( dt.getCategory(e18) == dt.getCategoryIdd("1") );

    Beliefs cwb = dt.getCategories(e15);
    BOOST_CHECK( cwb.size() > 0U );
    BOOST_CHECK_EQUAL( cwb[0].getValue()->get(), "1");
}

BOOST_AUTO_TEST_CASE( internalStateTest ) {

    DTC dt( createWeatherAttributes(), createWeatherCategory() );
	DecisionTreeTrainParam p = dt.getTrainParam();
	p.allowedNbrMiscEx = 0;
	dt.setTrainParam(p);
	dt.train( createWeatherTrainExamples(dt) );
	//cout << dt << endl;

	ostringstream oss;
	boost::archive::xml_oarchive oa(oss);

	oa << boost::serialization::make_nvp("DTC", dt );

	string dtcSerialString = oss.str();
	//cout << oss.str() << endl;

	{
		//check if restoring object from given string and serializing this restored object gives the string
		//of equal length
		std::istringstream iss(dtcSerialString);
		boost::archive::xml_iarchive ia(iss);

		DTC nnn;
		ia >> boost::serialization::make_nvp("DTC", nnn);
		//cout << nnn;
		ostringstream oss2;
		boost::archive::xml_oarchive oa2(oss2);

		oa2 << boost::serialization::make_nvp("DTC", nnn );

		BOOST_CHECK_EQUAL(dtcSerialString.size(), oss2.str().size() );
	}

	//setting
	DTC empty;

	std::istringstream iss(dtcSerialString);
	boost::archive::xml_iarchive ia(iss);
	ia >> boost::serialization::make_nvp("DTC", empty);
	//cout << empty << endl;

	string ET[] = { "overcast", "hot", "high", "weak"}; ExampleTest et = createExample( ET, ET + 4, empty);
	BOOST_CHECK_EQUAL( empty.getCategory(et), empty.getCategoryIdd("good") );

	string ET01[] = { "sunny", "hot", "high", "weak"};         ExampleTest et01 = createExample( ET01, ET01 + 4, empty);
	BOOST_CHECK( empty.getCategory(et01) == empty.getCategoryIdd("bad") );
    string ET02[] = { "sunny", "hot", "high", "strong"};       ExampleTest et02 = createExample( ET02, ET02 + 4, empty);
	BOOST_CHECK( empty.getCategory(et02) == empty.getCategoryIdd("bad") );
    string ET03[] = { "overcast", "hot", "high", "weak"};      ExampleTest et03 = createExample( ET03, ET03 + 4, empty);
	BOOST_CHECK( empty.getCategory(et03) == empty.getCategoryIdd("good") );
    string ET04[] = { "rain", "mild", "high", "weak"};         ExampleTest et04 = createExample( ET04, ET04 + 4, empty);
	BOOST_CHECK( empty.getCategory(et04) == empty.getCategoryIdd("good") );
    string ET05[] = { "rain", "cold", "normal", "weak"};       ExampleTest et05 = createExample( ET05, ET05 + 4, empty);
	BOOST_CHECK( empty.getCategory(et05) == empty.getCategoryIdd("good") );
    string ET06[] = { "rain", "cold", "normal", "strong"};     ExampleTest et06 = createExample( ET06, ET06 + 4, empty);
	BOOST_CHECK( empty.getCategory(et06) == empty.getCategoryIdd("bad") );
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

BOOST_AUTO_TEST_CASE( dtcValidator ) {
    DTC dt( createWeatherAttributes(), createWeatherCategory() );
    ExamplesTrain ex = createWeatherTrainExamples(dt);
    dt.train(ex);

    BOOST_CHECK( checkClassifier(ex, dt) >= static_cast<int>(ex.size()) - 2 );
    BOOST_CHECK( checkCross(ex, 14, dt) >= 0.5 );
}


BOOST_AUTO_TEST_CASE( pruneDTCTest ) {

    DTC dt( createWeatherAttributes(), createWeatherCategory() );
    dt.train( createWeatherTrainExamples(dt) );
    //cout << dt << endl;

    //new examples for pruning
    ExamplesTrain ex;
    string E01[] = { "sunny", "hot", "high", "weak"};         ex.push_back( createExample( E01, E01 + 4, "bad" , dt) );
    string E02[] = { "sunny", "hot", "high", "strong"};       ex.push_back( createExample( E02, E02 + 4, "bad" , dt) );
    string E03[] = { "overcast", "hot", "high", "weak"};      ex.push_back( createExample( E03, E03 + 4, "good", dt) );
    string E04[] = { "rain", "mild", "high", "weak"};         ex.push_back( createExample( E04, E04 + 4, "good", dt) );
    string E05[] = { "rain", "cold", "normal", "weak"};       ex.push_back( createExample( E05, E05 + 4, "good", dt) );
    string E06[] = { "rain", "cold", "normal", "strong"};     ex.push_back( createExample( E06, E06 + 4, "bad" , dt) );
    string E07[] = { "overcast", "cold", "normal", "strong"}; ex.push_back( createExample( E07, E07 + 4, "good", dt) );
    string E08[] = { "sunny", "mild", "high", "weak"};        ex.push_back( createExample( E08, E08 + 4, "bad" , dt) );
    string E09[] = { "sunny", "cold", "normal", "weak"};      ex.push_back( createExample( E09, E09 + 4, "good", dt) );
    string E10[] = { "rain", "mild", "normal", "weak"};       ex.push_back( createExample( E10, E10 + 4, "good", dt) );
    string E11[] = { "sunny", "mild", "normal", "strong"};    ex.push_back( createExample( E11, E11 + 4, "good", dt) );
    string E12[] = { "overcast", "mild", "high", "strong"};   ex.push_back( createExample( E12, E12 + 4, "good", dt) );
    string E13[] = { "overcast", "hot", "normal", "weak"};    ex.push_back( createExample( E13, E13 + 4, "good", dt) );
    string E14[] = { "rain", "mild", "high", "strong"};       ex.push_back( createExample( E14, E14 + 4, "bad" , dt) );
    // //new example - to make tests weaker and change internal node in tree into leaf node
    string E15[] = { "sunny", "cold", "high", "strong"};         ex.push_back( createExample( E15, E15 + 4, "good" , dt) );


    int bef_pruning = checkClassifier(ex, dt);

    dt.prune( ex );

    int after_pruning = checkClassifier(ex, dt);
    //pruning increase quality of classifier
    BOOST_CHECK( bef_pruning <= after_pruning);
    //cout << dt << endl;
}

BOOST_AUTO_TEST_SUITE_END()
