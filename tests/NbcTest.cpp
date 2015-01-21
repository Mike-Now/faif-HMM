/** plik zawiera test klasyfikatora */
#if defined (_MSC_VER) && (_MSC_VER >= 1400)
//msvc8.0 generuje smieci dla boost::string
#pragma warning(disable:4512)
#pragma warning(disable:4127)
//msvc9.0 warnings for boost::concept_check
#pragma warning(disable:4100)
//OK warning for boost::serializable (BOOST_CLASS_TRACKING), but not time now to resolve it correctly
#pragma warning(disable:4308)
#endif

#include <iostream>
#include <sstream>
#include <cassert>

#define BOOST_TEST_DYN_LINK
#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>

#include "../src/learning/Validator.hpp"
#include "../src/learning/NaiveBayesian.hpp"

using namespace std;
using namespace faif;
using namespace faif::ml;
using boost::unit_test::test_suite;

BOOST_AUTO_TEST_SUITE( FAIF_NBC_classifier_test )

typedef NaiveBayesian< ValueNominal<string> > NBC;
typedef NBC::AttrIdd AttrIdd;
typedef NBC::AttrDomain AttrDomain;
typedef NBC::Domains Domains;
typedef NBC::Beliefs Beliefs;
typedef NBC::ExampleTest ExampleTest;
typedef NBC::ExampleTrain ExampleTrain;
typedef NBC::ExamplesTrain ExamplesTrain;
int main(){}
BOOST_AUTO_TEST_CASE( classifierAddCategoryAttribTest ) {

    AttrDomain cat("category");
    cat.insert("one"); cat.insert("two"); cat.insert("three");

    AttrDomain a1("a");
    a1.insert("A"); a1.insert("B"); a1.insert("C");
    AttrDomain a2("i");
    a2.insert("I"); a2.insert("J");
    AttrDomain a3("p");
    a3.insert("P");

    Domains d;
    d.push_back(a1); d.push_back(a2); d.push_back(a3);

    NBC nb(d, cat);
    BOOST_CHECK( a1.getSize() == 3);
    BOOST_CHECK( a2.getSize() == 2);
    BOOST_CHECK( a3.getSize() == 1);
}

BOOST_AUTO_TEST_CASE( classifierCounterTest ) {
    const char* COUNTERS_OUTPUT_STRING =
        "Categories(3)=one,=two,=three,;\n"
        "Attributes(3):a=A,a=B,a=C,;i=I,i=J,;p=P,;\n"
        "State: TRAINING:\n"
        "=one(2):\n"
        "A(2),B(0),C(0),\n"
        "I(1),J(1),\n"
        "P(2),\n"
        "=two(1):\n"
        "A(0),B(1),C(0),\n"
        "I(1),J(0),\n"
        "P(1),\n"
        "=three(2):\n"
        "A(0),B(0),C(2),\n"
        "I(2),J(0),\n"
        "P(2),\n\n";

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

    NBC nb(d, cat);
    ExamplesTrain ex;
    string ex1[] = {"A","I","P"}; ex.push_back( createExample(ex1, ex1 + 3,"one", nb) );
    string ex2[] = {"A","J","P"}; ex.push_back( createExample(ex2, ex2 + 3,"one", nb) );
    string ex3[] = {"C","I","P"}; ex.push_back( createExample(ex3, ex3 + 3,"three", nb) );
    string ex4[] = {"C","I","P"}; ex.push_back( createExample(ex4, ex4 + 3,"three", nb) );
    string ex5[] = {"B","I","P"}; ex.push_back( createExample(ex5, ex5 + 3,"two", nb) );
    nb.train(ex);
    stringstream ss;
    ss << nb;
    BOOST_CHECK( ss.str() == string(COUNTERS_OUTPUT_STRING) );

    const char* RESET_OUTPUT_STRING =
        "Categories(3)=one,=two,=three,;\n"
        "Attributes(3):a=A,a=B,a=C,;i=I,i=J,;p=P,;\n"
        "State: TRAINING:\n"
        "=one(0):\n"
        "A(0),B(0),C(0),\n"
        "I(0),J(0),\n"
        "P(0),\n"
        "=two(0):\n"
        "A(0),B(0),C(0),\n"
        "I(0),J(0),\n"
        "P(0),\n"
        "=three(0):\n"
        "A(0),B(0),C(0),\n"
        "I(0),J(0),\n"
        "P(0),";

    nb.reset();
    stringstream st;
    st << nb;
    string s1 = st.str();
    boost::trim(s1);
    BOOST_CHECK( s1 == string(RESET_OUTPUT_STRING) );
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

    NBC nb(d, cat);
    ExamplesTrain ex;
    string ex1[] = {"A","I","P"}; ex.push_back( createExample(ex1, ex1 + 3,"one", nb) );
    string ex2[] = {"A","J","P"}; ex.push_back( createExample(ex2, ex2 + 3,"one", nb) );
    string ex3[] = {"C","I","P"}; ex.push_back( createExample(ex3, ex3 + 3,"three", nb) );
    string ex4[] = {"C","I","P"}; ex.push_back( createExample(ex4, ex4 + 3,"three", nb) );
	ex.push_back( createExample(ex4, ex4 + 3,"three", nb) );
    string ex5[] = {"B","I","P"}; ex.push_back( createExample(ex5, ex5 + 3,"two", nb) );
    nb.train(ex);

    string ex6e[] = {"B","J","P"};
    ExampleTest ex6 = createExample(ex6e, ex6e + 3, nb);
    AttrIdd c = nb.getCategory(ex6);
    BOOST_CHECK( c == nb.getCategoryIdd("two") );

    ExampleTest ex7; //empty example
    c = nb.getCategory(ex7);
    BOOST_CHECK( c == nb.getCategoryIdd("three") );
}

Domains createWeatherAttributes() {
    Domains attribs;
    string A1[] = {"slon", "poch", "desz" }; attribs.push_back( createDomain("aura", A1, A1 + 3) );
    string A2[] = {"cie", "umi", "zim"};     attribs.push_back( createDomain("temp", A2, A2 + 3) );
    string A3[] = {"norm", "duza"};          attribs.push_back( createDomain("wilg", A3, A3 + 2) );
    string A4[] = {"silny", "slaby"};        attribs.push_back( createDomain("wiatr", A4, A4 + 2) );
    return attribs;
}

AttrDomain createWeatherCategory() {
    string C[] = {"good","bad"};
    AttrDomain cat = createDomain("", C, C+2);
    return cat;
}

ExamplesTrain createWeatherTrainExamples(const NBC& nb) {
    ExamplesTrain ex;
    string e01[] = { "slon", "cie", "duza", "slaby"}; ex.push_back( createExample( e01, e01 + 4, "bad" , nb) );
    string e02[] = { "slon", "cie", "duza", "silny"}; ex.push_back( createExample( e02, e02 + 4, "bad" , nb) );
    string e03[] = { "poch", "cie", "duza", "slaby"}; ex.push_back( createExample( e03, e03 + 4, "good" , nb) );
    string e04[] = { "desz", "umi", "duza", "slaby"}; ex.push_back( createExample( e04, e04 + 4, "good" , nb) );
    string e05[] = { "desz", "zim", "norm", "slaby"}; ex.push_back( createExample( e05, e05 + 4, "good" , nb) );
    string e06[] = { "desz", "zim", "norm", "silny"}; ex.push_back( createExample( e06, e06 + 4, "bad" , nb) );
    string e07[] = { "poch", "zim", "norm", "silny"}; ex.push_back( createExample( e07, e07 + 4, "good" , nb) );
    string e08[] = { "slon", "umi", "duza", "slaby"}; ex.push_back( createExample( e08, e08 + 4, "bad" , nb) );
    string e09[] = { "slon", "zim", "norm", "slaby"}; ex.push_back( createExample( e09, e09 + 4, "good" , nb) );
    string e10[] = { "desz", "umi", "norm", "slaby"}; ex.push_back( createExample( e10, e10 + 4, "good" , nb) );
    string e11[] = { "slon", "umi", "norm", "silny"}; ex.push_back( createExample( e11, e11 + 4, "good" , nb) );
    string e12[] = { "poch", "umi", "duza", "silny"}; ex.push_back( createExample( e12, e12 + 4, "good" , nb) );
    string e13[] = { "poch", "cie", "norm", "slaby"}; ex.push_back( createExample( e13, e13 + 4, "good" , nb) );
    string e14[] = { "desz", "umi", "duza", "silny"}; ex.push_back( createExample( e14, e14 + 4, "bad" , nb) );
    return ex;
}

BOOST_AUTO_TEST_CASE( weatherClasifierTest ) {

    NBC n( createWeatherAttributes(), createWeatherCategory() );
    n.train( createWeatherTrainExamples(n) );

    string ET[] = { "slon", "cie", "duza", "slaby"};
    ExampleTest et = createExample( ET, ET + 4, n);
    BOOST_CHECK( n.getCategory(et) == n.getCategoryIdd("bad") );
    //std::cout << n << std::endl;
    string ET01[] = { "slon", "cie", "duza", "slaby"}; ExampleTest et01 = createExample( ET01, ET01 + 4, n);
    BOOST_CHECK( n.getCategory(et01) == n.getCategoryIdd("bad") );
    string ET02[] = { "slon", "cie", "duza", "silny"}; ExampleTest et02 = createExample( ET02, ET02 + 4, n);
    BOOST_CHECK( n.getCategory(et02) == n.getCategoryIdd("bad") );
    string ET03[] = { "poch", "cie", "duza", "slaby"}; ExampleTest et03 = createExample( ET03, ET03 + 4, n);
    BOOST_CHECK( n.getCategory(et03) == n.getCategoryIdd("good") );
    string ET04[] = { "desz", "umi", "duza", "slaby"}; ExampleTest et04 = createExample( ET04, ET04 + 4, n);
    BOOST_CHECK( n.getCategory(et04) == n.getCategoryIdd("good") );
    string ET05[] = { "desz", "zim", "norm", "slaby"}; ExampleTest et05 = createExample( ET05, ET05 + 4, n);
    BOOST_CHECK( n.getCategory(et05) == n.getCategoryIdd("good") );
    // string ET06[] = { "desz", "zim", "norm", "silny"}; ExampleTest et06 = createExample( ET06, ET06 + 4, n);
    // BOOST_CHECK( n.getCategory(et06) == n.getCategoryIdd("bad") );
    string ET07[] = { "poch", "zim", "norm", "silny"}; ExampleTest et07 = createExample( ET07, ET07 + 4, n);
    BOOST_CHECK( n.getCategory(et07) == n.getCategoryIdd("good") );
    string ET08[] = { "slon", "umi", "duza", "slaby"}; ExampleTest et08 = createExample( ET08, ET08 + 4, n);
    BOOST_CHECK( n.getCategory(et08) == n.getCategoryIdd("bad") );
    string ET09[] = { "slon", "zim", "norm", "slaby"}; ExampleTest et09 = createExample( ET09, ET09 + 4, n);
    BOOST_CHECK( n.getCategory(et09) == n.getCategoryIdd("good") );
    string ET10[] = { "desz", "umi", "norm", "slaby"}; ExampleTest et10 = createExample( ET10, ET10 + 4, n);
    BOOST_CHECK( n.getCategory(et10) == n.getCategoryIdd("good") );
    string ET11[] = { "slon", "umi", "norm", "silny"}; ExampleTest et11 = createExample( ET11, ET11 + 4, n);
    BOOST_CHECK( n.getCategory(et11) == n.getCategoryIdd("good") );
    string ET12[] = { "poch", "umi", "duza", "silny"}; ExampleTest et12 = createExample( ET12, ET12 + 4, n);
    BOOST_CHECK( n.getCategory(et12) == n.getCategoryIdd("good") );
    string ET13[] = { "poch", "cie", "norm", "slaby"}; ExampleTest et13 = createExample( ET13, ET13 + 4, n);
    BOOST_CHECK( n.getCategory(et13) == n.getCategoryIdd("good") );
    string ET14[] = { "desz", "umi", "duza", "silny"}; ExampleTest et14 = createExample( ET14, ET14 + 4, n);
    BOOST_CHECK( n.getCategory(et14) == n.getCategoryIdd("bad") );

}

BOOST_AUTO_TEST_CASE( weatherClasifierGetCategories ) {

    NBC n( createWeatherAttributes(), createWeatherCategory() );
    n.train( createWeatherTrainExamples(n) );

    string ET[] = { "slon", "cie", "duza", "slaby"};
    ExampleTest et = createExample( ET, ET + 4, n);
    Beliefs cwb = n.getCategories(et);
    BOOST_CHECK_EQUAL( cwb.size(), 2U );
    //the best has index 0
    BOOST_CHECK_EQUAL( cwb[0].getValue()->get(), "bad");
    BOOST_CHECK_CLOSE_FRACTION( cwb[0].getProbability(), 0.7042, 0.001 );
    BOOST_CHECK_EQUAL( cwb[1].getValue()->get(), "good");
    BOOST_CHECK_CLOSE_FRACTION( cwb[1].getProbability(), 0.2958, 0.001 );
    //check the same again-but now classifier is switched
    cwb = n.getCategories(et);
    BOOST_CHECK_EQUAL( cwb.size(), 2U );
    BOOST_CHECK_EQUAL( cwb[0].getValue()->get(), "bad");
    BOOST_CHECK_CLOSE_FRACTION( cwb[0].getProbability(), 0.7042, 0.001 );
    BOOST_CHECK_EQUAL( cwb[1].getValue()->get(), "good");
    BOOST_CHECK_CLOSE_FRACTION( cwb[1].getProbability(), 0.2958, 0.001 );
}

BOOST_AUTO_TEST_CASE( internalStateTest ) {
    NBC n( createWeatherAttributes(), createWeatherCategory() );
    n.train( createWeatherTrainExamples(n) );
    const char* TRAINED_NBC_STRING =
        "Categories(2)=good,=bad,;\n"
        "Attributes(4):aura=slon,aura=poch,aura=desz,;temp=cie,temp=umi,temp=zim,;wilg=norm,wilg=duza,;wiatr=silny,wiatr=slaby,;\n"
        "State: TRAINING:\n"
        "=good(9):\n"
        "slon(2),poch(4),desz(3),\n"
        "cie(2),umi(4),zim(3),\n"
        "norm(6),duza(3),\n"
        "silny(3),slaby(6),\n"
        "=bad(5):\n"
        "slon(3),poch(0),desz(2),\n"
        "cie(2),umi(2),zim(1),\n"
        "norm(1),duza(4),\n"
        "silny(3),slaby(2),";
    BOOST_CHECK( boost::trim_copy(boost::lexical_cast<string>(n)) == string(TRAINED_NBC_STRING) );

    ostringstream oss;
    boost::archive::xml_oarchive oa(oss);

    oa << boost::serialization::make_nvp("NBC", n );

    string nbcSerialString = oss.str();
    //cout << oss.str() << endl;

    //setting
    NBC emptyNb( createWeatherAttributes(), createWeatherCategory() );
    const char* EMPTY_NBC_STRING =
        "Categories(2)=good,=bad,;\n"
        "Attributes(4):aura=slon,aura=poch,aura=desz,;temp=cie,temp=umi,temp=zim,;wilg=norm,wilg=duza,;wiatr=silny,wiatr=slaby,;\n"
        "State: TRAINING:\n"
        "=good(0):\n"
        "slon(0),poch(0),desz(0),\n"
        "cie(0),umi(0),zim(0),\n"
        "norm(0),duza(0),\n"
        "silny(0),slaby(0),\n"
        "=bad(0):\n"
        "slon(0),poch(0),desz(0),\n"
        "cie(0),umi(0),zim(0),\n"
        "norm(0),duza(0),\n"
        "silny(0),slaby(0),";
    BOOST_CHECK( boost::trim_copy(boost::lexical_cast<string>(emptyNb)) == string(EMPTY_NBC_STRING) );

    {
        //check if restoring object from given string and serializing this restored object gives the string
        //of equal length
        std::istringstream iss(nbcSerialString);
        boost::archive::xml_iarchive ia(iss);

        NBC nnn;
        ia >> boost::serialization::make_nvp("NBC", nnn);

        ostringstream oss2;
        boost::archive::xml_oarchive oa2(oss2);

        oa2 << boost::serialization::make_nvp("NBC", nnn );

        BOOST_CHECK_EQUAL(nbcSerialString.size(), oss2.str().size() );
    }

    std::istringstream iss(nbcSerialString);
    boost::archive::xml_iarchive ia(iss);
    ia >> boost::serialization::make_nvp("NBC", emptyNb);
    //cout << emptyNb << endl;

    string ET01[] = { "slon", "cie", "duza", "slaby"}; ExampleTest et01 = createExample( ET01, ET01 + 4, emptyNb);
    BOOST_CHECK( emptyNb.getCategory(et01) == emptyNb.getCategoryIdd("bad") );
    string ET02[] = { "slon", "cie", "duza", "silny"}; ExampleTest et02 = createExample( ET02, ET02 + 4, emptyNb);
    BOOST_CHECK( emptyNb.getCategory(et02) == emptyNb.getCategoryIdd("bad") );
    string ET03[] = { "poch", "cie", "duza", "slaby"}; ExampleTest et03 = createExample( ET03, ET03 + 4, emptyNb);
    BOOST_CHECK( emptyNb.getCategory(et03) == emptyNb.getCategoryIdd("good") );
    string ET04[] = { "desz", "umi", "duza", "slaby"}; ExampleTest et04 = createExample( ET04, ET04 + 4, emptyNb);
    BOOST_CHECK( emptyNb.getCategory(et04) == emptyNb.getCategoryIdd("good") );
    string ET05[] = { "desz", "zim", "norm", "slaby"}; ExampleTest et05 = createExample( ET05, ET05 + 4, emptyNb);
    BOOST_CHECK( emptyNb.getCategory(et05) == emptyNb.getCategoryIdd("good") );
    // string ET06[] = { "desz", "zim", "norm", "silny"}; ExampleTest et06 = createExample( ET06, ET06 + 4, emptyNb);
    // BOOST_CHECK( emptyNb.getCategory(et06) == emptyNb.getCategoryIdd("bad") );
    string ET07[] = { "poch", "zim", "norm", "silny"}; ExampleTest et07 = createExample( ET07, ET07 + 4, emptyNb);
    BOOST_CHECK( emptyNb.getCategory(et07) == emptyNb.getCategoryIdd("good") );
    string ET08[] = { "slon", "umi", "duza", "slaby"}; ExampleTest et08 = createExample( ET08, ET08 + 4, emptyNb);
    BOOST_CHECK( emptyNb.getCategory(et08) == emptyNb.getCategoryIdd("bad") );
    string ET09[] = { "slon", "zim", "norm", "slaby"}; ExampleTest et09 = createExample( ET09, ET09 + 4, emptyNb);
    BOOST_CHECK( emptyNb.getCategory(et09) == emptyNb.getCategoryIdd("good") );
    string ET10[] = { "desz", "umi", "norm", "slaby"}; ExampleTest et10 = createExample( ET10, ET10 + 4, emptyNb);
    BOOST_CHECK( emptyNb.getCategory(et10) == emptyNb.getCategoryIdd("good") );
    string ET11[] = { "slon", "umi", "norm", "silny"}; ExampleTest et11 = createExample( ET11, ET11 + 4, emptyNb);
    BOOST_CHECK( emptyNb.getCategory(et11) == emptyNb.getCategoryIdd("good") );
    string ET12[] = { "poch", "umi", "duza", "silny"}; ExampleTest et12 = createExample( ET12, ET12 + 4, emptyNb);
    BOOST_CHECK( emptyNb.getCategory(et12) == emptyNb.getCategoryIdd("good") );
    string ET13[] = { "poch", "cie", "norm", "slaby"}; ExampleTest et13 = createExample( ET13, ET13 + 4, emptyNb);
    BOOST_CHECK( emptyNb.getCategory(et13) == emptyNb.getCategoryIdd("good") );
    string ET14[] = { "desz", "umi", "duza", "silny"}; ExampleTest et14 = createExample( ET14, ET14 + 4, emptyNb);
    BOOST_CHECK( emptyNb.getCategory(et14) == emptyNb.getCategoryIdd("bad") );
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

    NBC n(attribs, cat);

    //funkcja wynikowa to: c1 <=> (wyk = ww) or (wyk = ws) and (wi = w1) or (wyk = wp) and (wi = w1) and (plec = k)
    ExamplesTrain ex;
    string e01S[] = { "pod", "mlody", "kob" };   ExampleTrain e01 = createExample(e01S, e01S+3, "1" , n); n.trainIncremental(e01);
    string e02S[] = { "pod", "mlody", "mez" };   ExampleTrain e02 = createExample(e02S, e02S+3, "0" , n); n.trainIncremental(e02);
    string e03S[] = { "pod", "sredn", "kob" };   ExampleTrain e03 = createExample(e03S, e03S+3, "0" , n); // n.trainIncremental(e03);
    string e04S[] = { "pod", "sredn", "mez" };   ExampleTrain e04 = createExample(e04S, e04S+3, "0" , n); n.trainIncremental(e04);
    string e05S[] = { "pod", "stary", "kob" };   ExampleTrain e05 = createExample(e05S, e05S+3, "0" , n); n.trainIncremental(e05);
    string e06S[] = { "pod", "stary", "mez" };   ExampleTrain e06 = createExample(e06S, e06S+3, "0" , n); n.trainIncremental(e06);
    string e07S[] = { "sre", "mlody", "kob" };   ExampleTrain e07 = createExample(e07S, e07S+3, "1" , n); n.trainIncremental(e07);
    string e08S[] = { "sre", "mlody", "mez" };   ExampleTrain e08 = createExample(e08S, e08S+3, "1" , n); n.trainIncremental(e08);
    string e09S[] = { "sre", "sredn", "kob" };   ExampleTrain e09 = createExample(e09S, e09S+3, "0" , n); n.trainIncremental(e09);
    string e10S[] = { "sre", "sredn", "mez" };   ExampleTrain e10 = createExample(e10S, e10S+3, "0" , n); n.trainIncremental(e10);
    string e11S[] = { "sre", "stary", "kob" };   ExampleTrain e11 = createExample(e11S, e11S+3, "0" , n); // n.trainIncremental(e11);
    string e12S[] = { "sre", "stary", "mez" };   ExampleTrain e12 = createExample(e12S, e12S+3, "0" , n); n.trainIncremental(e12);
    string e13S[] = { "wyz", "mlody", "kob" };   ExampleTrain e13 = createExample(e13S, e13S+3, "1" , n); // n.trainIncremental(e13);
    string e14S[] = { "wyz", "mlody", "mez" };   ExampleTrain e14 = createExample(e14S, e14S+3, "1" , n); n.trainIncremental(e14);
    string e15S[] = { "wyz", "sredn", "kob" };   ExampleTrain e15 = createExample(e15S, e15S+3, "1" , n); // n.trainIncremental(e15);
    string e16S[] = { "wyz", "sredn", "mez" };   ExampleTrain e16 = createExample(e16S, e16S+3, "1" , n); n.trainIncremental(e16);
    string e17S[] = { "wyz", "stary", "kob" };   ExampleTrain e17 = createExample(e17S, e17S+3, "1" , n); n.trainIncremental(e17);
    string e18S[] = { "wyz", "stary", "mez" };   ExampleTrain e18 = createExample(e18S, e18S+3, "1" , n); n.trainIncremental(e18);

    BOOST_CHECK( n.getCategory(e01) == n.getCategoryIdd("1") );
    BOOST_CHECK( n.getCategory(e02) == n.getCategoryIdd("0") );
    BOOST_CHECK( n.getCategory(e03) == n.getCategoryIdd("0") );
    BOOST_CHECK( n.getCategory(e04) == n.getCategoryIdd("0") );
    BOOST_CHECK( n.getCategory(e05) == n.getCategoryIdd("0") );
    BOOST_CHECK( n.getCategory(e06) == n.getCategoryIdd("0") );
    BOOST_CHECK( n.getCategory(e07) == n.getCategoryIdd("1") );
    BOOST_CHECK( n.getCategory(e08) == n.getCategoryIdd("1") );
    BOOST_CHECK( n.getCategory(e09) == n.getCategoryIdd("0") );
    BOOST_CHECK( n.getCategory(e10) == n.getCategoryIdd("0") );
    BOOST_CHECK( n.getCategory(e11) == n.getCategoryIdd("0") );
    BOOST_CHECK( n.getCategory(e12) == n.getCategoryIdd("0") );
    BOOST_CHECK( n.getCategory(e13) == n.getCategoryIdd("1") );
    BOOST_CHECK( n.getCategory(e14) == n.getCategoryIdd("1") );
    BOOST_CHECK( n.getCategory(e15) == n.getCategoryIdd("1") );
    BOOST_CHECK( n.getCategory(e16) == n.getCategoryIdd("1") );
    BOOST_CHECK( n.getCategory(e17) == n.getCategoryIdd("1") );
    BOOST_CHECK( n.getCategory(e18) == n.getCategoryIdd("1") );

    Beliefs cwb = n.getCategories(e15);
    BOOST_CHECK_EQUAL( cwb.size(), 2U );
    BOOST_CHECK_EQUAL( cwb[0].getValue()->get(), "1");
    BOOST_CHECK_CLOSE_FRACTION( cwb[0].getProbability(), 0.7692, 0.001 );
    BOOST_CHECK_EQUAL( cwb[1].getValue()->get(), "0");
    BOOST_CHECK_CLOSE_FRACTION( cwb[1].getProbability(), 0.2308, 0.001 );
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

    NBC n(attribs, cat);

    string e01S[] = { "pod", "mlody", "kob" };   ExampleTrain e01 = createExample(e01S, e01S+3, "1" , n); n.trainIncremental(e01);
    string e02S[] = { "pod", "mlody", "mez" };   ExampleTrain e02 = createExample(e02S, e02S+3, "2" , n); n.trainIncremental(e02);
    string e03S[] = { "pod", "sredn", "kob" };   ExampleTrain e03 = createExample(e03S, e03S+3, "2" , n); // n.trainIncremental(e03);
    string e04S[] = { "pod", "sredn", "mez" };   ExampleTrain e04 = createExample(e04S, e04S+3, "2" , n); n.trainIncremental(e04);
    string e05S[] = { "pod", "stary", "kob" };   ExampleTrain e05 = createExample(e05S, e05S+3, "2" , n); n.trainIncremental(e05);
    string e06S[] = { "pod", "stary", "mez" };   ExampleTrain e06 = createExample(e06S, e06S+3, "2" , n); n.trainIncremental(e06);
    string e07S[] = { "sre", "mlody", "kob" };   ExampleTrain e07 = createExample(e07S, e07S+3, "1" , n); n.trainIncremental(e07);
    string e08S[] = { "sre", "mlody", "mez" };   ExampleTrain e08 = createExample(e08S, e08S+3, "1" , n); n.trainIncremental(e08);
    string e09S[] = { "sre", "sredn", "kob" };   ExampleTrain e09 = createExample(e09S, e09S+3, "0" , n); n.trainIncremental(e09);
    string e10S[] = { "sre", "sredn", "mez" };   ExampleTrain e10 = createExample(e10S, e10S+3, "0" , n); n.trainIncremental(e10);
    string e11S[] = { "sre", "stary", "kob" };   ExampleTrain e11 = createExample(e11S, e11S+3, "0" , n); // n.trainIncremental(e11);
    string e12S[] = { "sre", "stary", "mez" };   ExampleTrain e12 = createExample(e12S, e12S+3, "0" , n); n.trainIncremental(e12);
    string e13S[] = { "wyz", "mlody", "kob" };   ExampleTrain e13 = createExample(e13S, e13S+3, "1" , n); // n.trainIncremental(e13);
    string e14S[] = { "wyz", "mlody", "mez" };   ExampleTrain e14 = createExample(e14S, e14S+3, "1" , n); n.trainIncremental(e14);
    string e15S[] = { "wyz", "sredn", "kob" };   ExampleTrain e15 = createExample(e15S, e15S+3, "1" , n); // n.trainIncremental(e15);
    string e16S[] = { "wyz", "sredn", "mez" };   ExampleTrain e16 = createExample(e16S, e16S+3, "1" , n); n.trainIncremental(e16);
    string e17S[] = { "wyz", "stary", "kob" };   ExampleTrain e17 = createExample(e17S, e17S+3, "1" , n); n.trainIncremental(e17);
    string e18S[] = { "wyz", "stary", "mez" };   ExampleTrain e18 = createExample(e18S, e18S+3, "1" , n); n.trainIncremental(e18);

    BOOST_CHECK( n.getCategory(e01) == n.getCategoryIdd("1") );
    BOOST_CHECK( n.getCategory(e02) == n.getCategoryIdd("2") );
    BOOST_CHECK( n.getCategory(e03) == n.getCategoryIdd("2") );
    BOOST_CHECK( n.getCategory(e04) == n.getCategoryIdd("2") );
    BOOST_CHECK( n.getCategory(e05) == n.getCategoryIdd("2") );
    BOOST_CHECK( n.getCategory(e06) == n.getCategoryIdd("2") );
    BOOST_CHECK( n.getCategory(e07) == n.getCategoryIdd("1") );
    BOOST_CHECK( n.getCategory(e08) == n.getCategoryIdd("1") );
    BOOST_CHECK( n.getCategory(e09) == n.getCategoryIdd("0") );
    BOOST_CHECK( n.getCategory(e10) == n.getCategoryIdd("0") );
    BOOST_CHECK( n.getCategory(e11) == n.getCategoryIdd("0") );
    BOOST_CHECK( n.getCategory(e12) == n.getCategoryIdd("0") );
    BOOST_CHECK( n.getCategory(e13) == n.getCategoryIdd("1") );
    BOOST_CHECK( n.getCategory(e14) == n.getCategoryIdd("1") );
    BOOST_CHECK( n.getCategory(e15) == n.getCategoryIdd("1") );
    BOOST_CHECK( n.getCategory(e16) == n.getCategoryIdd("1") );
    BOOST_CHECK( n.getCategory(e17) == n.getCategoryIdd("1") );
    BOOST_CHECK( n.getCategory(e18) == n.getCategoryIdd("1") );

    Beliefs cwb = n.getCategories(e15);
    BOOST_CHECK_EQUAL( cwb.size(), 3U );
    BOOST_CHECK_EQUAL( cwb[0].getValue()->get(), "1");
    BOOST_CHECK_CLOSE_FRACTION( cwb[0].getProbability(), 0.6384, 0.001 );
    BOOST_CHECK_EQUAL( cwb[1].getValue()->get(), "0");
    BOOST_CHECK_CLOSE_FRACTION( cwb[1].getProbability(), 0.2394, 0.001 );
    BOOST_CHECK_EQUAL( cwb[2].getValue()->get(), "2");
    BOOST_CHECK_CLOSE_FRACTION( cwb[2].getProbability(), 0.1222, 0.001 );
}

BOOST_AUTO_TEST_CASE( nbcValidator ) {

    NBC n( createWeatherAttributes(), createWeatherCategory() );
    ExamplesTrain ex = createWeatherTrainExamples(n);
    n.train(ex);

    BOOST_CHECK( checkClassifier(ex, n) >= static_cast<int>(ex.size()) - 2 );
    BOOST_CHECK( checkCross(ex, 14, n) >= 0.5 );
}

BOOST_AUTO_TEST_SUITE_END()

