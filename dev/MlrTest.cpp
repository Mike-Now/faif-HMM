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
#include <fstream>
#include <cmath>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE MLR_suite
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include "Validator.hpp" //TODO
#include "MLReg.hpp" //TODO

using namespace std;
using namespace faif;
using namespace faif::ml;
using boost::unit_test::test_suite;

BOOST_AUTO_TEST_SUITE( FAIF_MLR_classifier_test )

typedef MLReg< ValueNominal<string> > MLR;
typedef MLR::AttrIdd AttrIdd;
typedef MLR::AttrDomain AttrDomain;
typedef MLR::Domains Domains;
typedef MLR::Beliefs Beliefs;
typedef MLR::ExampleTest ExampleTest;
typedef MLR::ExampleTrain ExampleTrain;
typedef MLR::ExamplesTrain ExamplesTrain;
typedef MLR::NormExample NormExample;
typedef MLR::NormExamples NormExamples;
typedef MLR::NCategoryId NCategoryId;
typedef MLR::Matrix Matrix;
typedef MLR::Vector Vector;
 /* int main(){ */
	/* faif::ml::MLReg<faif::ValueNominal<std::string> > test; */
	/* faif::ml::MLReg<faif::ValueNominal<std::string> > test2; */

	/* { */
	/* 	std::ofstream ofs("serial"); */

	/* 	boost::archive::text_oarchive oa(ofs); */

	/* 	oa << test; */
	/* } */
	/* { */
	/* 	std::ifstream ifs("serial"); */

	/* 	boost::archive::text_iarchive ia(ifs); */

	/* 	ia >> test2; */
	/* } */

    /* std::cout<<"All tests for Multinomial Logistic Regression Classifier passed."<<std::endl; */

    /* return 0; */
/* } */

double calcExp(NormExample &ex, Matrix&mat,int row);
double calcExp(NormExample &ex, Matrix&mat,int row)
{
    double power=0;
    for(int i=0;i<ex.size();i++){

        power=power+ex[i]*mat[row][i];
    }
    return std::exp(power);

}

BOOST_AUTO_TEST_CASE(SoftMaxFunTest){

    NormExample ex(3);
    NCategoryId catId = 0;
    Matrix mat(3,3);
    mat[0][0]=2;mat[0][1]=5;mat[0][2]=2;
    mat[1][0]=0;mat[1][1]=2;mat[1][2]=4;
    mat[2][0]=2;mat[2][1]=7;mat[2][2]=9;

    ex[0]=3;ex[1]=10;ex[2]=15;

    double assertProb = calcExp(ex,mat,catId);
    double denom=0;
    for(int i=0;i<3;i++){
        denom+=calcExp(ex,mat,i);
    }
    assertProb=assertProb/denom;
    double prob = MLR::calcSoftMax(ex,catId,mat);
    BOOST_CHECK_EQUAL(prob,assertProb);
}


BOOST_AUTO_TEST_CASE(CalcGradTest){
    Matrix parameters(3,3);

    NormExample tEx(3);
    tEx[0]=5;tEx[1]=2;tEx[2]=9;
    tEx.category=0;
    NormExamples tExamples(3,3,1);
    tExamples[0]=tEx;
    NCategoryId catId=0;

    Vector grad = MLR::BGDTraining::calcGrad(tExamples,parameters,catId);
    Vector assertGrad(3);
    double softMaxVal = MLR::calcSoftMax(tEx,tEx.category,parameters);
    assertGrad[0]=(tEx[0] * (1.0-softMaxVal))/-1;
    assertGrad[1]=(tEx[1] * (1.0-softMaxVal))/-1;
    assertGrad[2]=(tEx[2] * (1.0-softMaxVal))/-1;
    BOOST_CHECK_EQUAL_COLLECTIONS(grad.origin(),grad.origin()+grad.num_elements(),assertGrad.origin(),assertGrad.origin()+assertGrad.num_elements());

    for(int i=0;i<assertGrad.size();i++)
    {
        parameters[0][i]=parameters[0][i]-assertGrad[i];
    }

    catId=1;

    grad = MLR::BGDTraining::calcGrad(tExamples,parameters,catId);

    softMaxVal = MLR::calcSoftMax(tEx,tEx.category,parameters);
    assertGrad[0]=(tEx[0] * (0.0-softMaxVal))/-1;
    assertGrad[1]=(tEx[1] * (0.0-softMaxVal))/-1;
    assertGrad[2]=(tEx[2] * (0.0-softMaxVal))/-1;
    BOOST_CHECK_EQUAL_COLLECTIONS(grad.origin(),grad.origin()+grad.num_elements(),assertGrad.origin(),assertGrad.origin()+assertGrad.num_elements());

}

Domains createWeatherAttributes();
Domains createWeatherAttributes() {
    Domains attribs;
    string A1[] = {"slon", "poch", "desz" }; attribs.push_back( createDomain("aura", A1, A1 + 3) );
    string A2[] = {"cie", "umi", "zim"};     attribs.push_back( createDomain("temp", A2, A2 + 3) );
    string A3[] = {"norm", "duza"};          attribs.push_back( createDomain("wilg", A3, A3 + 2) );
    string A4[] = {"silny", "slaby"};        attribs.push_back( createDomain("wiatr", A4, A4 + 2) );
    return attribs;
}

AttrDomain createWeatherCategory();
AttrDomain createWeatherCategory() {
    string C[] = {"good","bad"};
    AttrDomain cat = createDomain("", C, C+2);
    return cat;
}
ExamplesTrain createWeatherTrainExamples(const MLR& nb);
ExamplesTrain createWeatherTrainExamples(const MLR& nb) {
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

    MLR n( createWeatherAttributes(), createWeatherCategory(),"BGD" );
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
BOOST_AUTO_TEST_SUITE_END()

