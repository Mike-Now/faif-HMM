/** the learning primitives tests (attribute, example, validator, fusion) */

#if defined(_MSC_VER)
/** annoying warning msvc 'std::copy is deprecated' */
#pragma warning(disable:4996)
//warning msvc dla boost::random
#pragma warning(disable:4100)
#pragma warning(disable:4512)
#pragma warning(disable:4127)
//OK warning for boost::serializable (BOOST_CLASS_TRACKING), but not time now to resolve it correctly
#pragma warning(disable:4308)
#endif

#include <cmath>
#include <ctime>
#include <algorithm>
#include <functional>
#include <boost/random.hpp>
#include <boost/bind.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>

#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include "../src/learning/Classifier.hpp"
#include "../src/learning/Validator.hpp"
#include "../src/learning/Fusion.hpp"


using namespace std;
using namespace faif;
using namespace faif::ml;
using boost::unit_test::test_suite;

BOOST_AUTO_TEST_SUITE( learning_test )

BOOST_AUTO_TEST_CASE( beliefTest ) {

    typedef Belief<ValueNominal<string> > Bel;
    typedef Bel::Value::DomainType AttrDomain;
    typedef Bel::ValueId AttrIdd;
    typedef Bel::Beliefs Beliefs;

    AttrDomain a("x");
    AttrIdd v1 = a.insert("A");
    AttrIdd v2 = a.insert("B");

    Beliefs cwb;
    cwb.push_back( Bel(v1,0.3) );
    cwb.push_back( Bel(v2,0.7) );

    BOOST_CHECK_EQUAL( cwb[0].getValue(), v1 );
    BOOST_CHECK_EQUAL( cwb[0].getProbability(), 0.3 );
    BOOST_CHECK_EQUAL( boost::lexical_cast<string>(cwb[0]), string("Value:x=A Prob:0.3") );
    BOOST_CHECK_EQUAL( boost::lexical_cast<string>(cwb), string("Value:x=A Prob:0.3;Value:x=B Prob:0.7;") );
}

BOOST_AUTO_TEST_CASE( classifierExampleTest ) {

    typedef Classifier<ValueNominal<string> > Classifier;
    typedef Classifier::AttrDomain AttrDomain;
    typedef Classifier::AttrIdd AttrIdd;
    typedef Classifier::ExampleTest ExampleTest;
    typedef Classifier::ExampleTrain ExampleTrain;

    AttrDomain a("x");
    AttrIdd va1 = a.insert("A");
    AttrIdd va2 = a.insert("B");
    AttrIdd va3 = a.insert("C");

    ExampleTest exx1;
    BOOST_CHECK( exx1.size() == 0 );
    exx1.push_back(va1); exx1.push_back(va2); exx1.push_back(va3);
    BOOST_CHECK( exx1.size() == 3 );
    ExampleTest exx2 (exx1);
    BOOST_CHECK( exx1 == exx2 );
    exx2.push_back(va1); exx2.push_back(va1);
    BOOST_CHECK( exx2.size() == 5 );
    BOOST_CHECK( exx1 != exx2 );
    ExampleTest exx3 = exx1;
    BOOST_CHECK( exx1 == exx3 );
    exx3 = exx2;
    BOOST_CHECK( exx1 != exx3 );
    BOOST_CHECK( exx2 == exx3 );

    stringstream ss;
    ss << exx1;
    BOOST_CHECK( ss.str() == std::string("x=A x=B x=C ") );

    ExampleTrain ex1;
    BOOST_CHECK( ex1.size() == 0 );
    BOOST_CHECK( ex1.getFeature() == AttrDomain::getUnknownId() );

    ExampleTrain ex2(exx1, va3);
    BOOST_CHECK( ex2.size() == 3 );
    BOOST_CHECK( ex2.getFeature() == va3 );
    BOOST_CHECK( ex2 == exx1 );

    ExampleTrain ex3(ex2);
    BOOST_CHECK( ex3 == ex2 );
    BOOST_CHECK( ex3 != ex1 );

    ex3 = ex1;
    BOOST_CHECK( ex3 != ex2 );
    BOOST_CHECK( ex3 == ex1 );

    stringstream st;
    st << ex2;
    BOOST_CHECK( st.str() == std::string("x=A x=B x=C : x=C") );
}

BOOST_AUTO_TEST_CASE( exampleShortConstructTest ) {

    typedef Classifier<ValueNominal<string> > Classifier;
    typedef Classifier::AttrDomain AttrDomain;
    typedef Classifier::AttrIdd AttrIdd;
    typedef Classifier::ExampleTest ExampleTest;
    typedef Classifier::ExampleTrain ExampleTrain;

    AttrDomain a("x");
    AttrIdd va1 = a.insert("A");

    ExampleTest ex1; ex1.push_back(va1);
    BOOST_CHECK( ex1.size() == 1 );

    ex1.push_back(va1);
    ex1.push_back(va1);

    ExampleTrain e1( ex1, va1 );
    BOOST_CHECK( e1.getFeature() == va1 );
    BOOST_CHECK( e1.size() == 3 );
}

namespace {

    /** Example classifier, to test the classifier interface.
        This class counts the number of training examples.
    */
    class DummyCountClassifier
        : public Classifier<ValueNominal<string> > {

    public:
        DummyCountClassifier() : Classifier(), count_(0) {}

        DummyCountClassifier(const Domains& attr, const AttrDomain& category)
            : Classifier(attr, category), count_(0) {
        }

        /** switch the internal classifier state to initial (not learning anything) */
        virtual void reset() { count_ = 0; }

        /** add the one training example (incremental learning) */
        virtual void train(const ExamplesTrain& c) { count_ += c.size(); }

        /** required method */
        virtual AttrIdd getCategory(const ExampleTest&) const { return AttrDomain::getUnknownId(); }

        /** required method */
        virtual Beliefs getCategories(const ExampleTest& example) const { return Beliefs(); }

        // return the counter
        int getCount() const { return count_; }
    private:
        int count_;
    };

} //namespace

BOOST_AUTO_TEST_CASE( exampleCreationTest ) {

    typedef Classifier<ValueNominal<string> > Classifier;
    typedef Classifier::AttrDomain AttrDomain;
    typedef Classifier::Domains Domains;
    typedef Classifier::ExampleTest ExampleTest;
    typedef Classifier::ExampleTrain ExampleTrain;

    string C[] = {"0","1"};
    AttrDomain cat = createDomain("c", C, C+2);
    Domains domains;
    string A[] = {"A", "B", "C" };
    domains.push_back( createDomain("x", A, A + 3) );
    domains.push_back( createDomain("y", A, A + 3) );

    DummyCountClassifier classifier(domains, cat);

    string ET[] = { "A", "A"};
    ExampleTest et = createExample( ET, ET + 2, classifier);
    ExampleTrain etr = createExample( ET, ET + 2, "0", classifier);

    BOOST_CHECK( boost::lexical_cast<string>(et) == string("x=A y=A ") );
    BOOST_CHECK( boost::lexical_cast<string>(etr) == string("x=A y=A : c=0") );

    //the creator from pairs <categ_name, value>
    vector<pair<string,string> > v;
    v.push_back( make_pair<string,string>("y","B") ); //the different sequence
    v.push_back( make_pair<string,string>("x","A") );

    ExampleTest et1 = createExample(v, classifier);
    ExampleTrain etr1 = createExample(v, "0", classifier);
    BOOST_CHECK( boost::lexical_cast<string>(et1) == string("x=A y=B ") );
    BOOST_CHECK( boost::lexical_cast<string>(etr1) == string("x=A y=B : c=0") );

    vector<pair<string,string> > v2;
    //empty example test
    et1 = createExample(v2, classifier);
    //cout << et1 << endl;
    BOOST_CHECK( et1.size() == 2U);

    v2.push_back( make_pair<string,string>("y","B") ); //the different sequence
    //no value for x
    BOOST_CHECK_THROW( createExampleStrict(v2, classifier), NotFoundException );
}

BOOST_AUTO_TEST_CASE( exampleMajorCategory ) {

    typedef Classifier<ValueNominal<string> > Classifier;
    typedef Classifier::AttrDomain AttrDomain;
    typedef Classifier::Domains Domains;
    typedef Classifier::ExamplesTrain ExamplesTrain;

    AttrDomain cat("c");
    cat.insert("0");
    cat.insert("1");

    Domains domains;
    string A[] = {"A", "B", "C" };
    domains.push_back( createDomain("x", A, A + 3) );
    domains.push_back( createDomain("y", A, A + 3) );

    DummyCountClassifier classifier(domains, cat);

    string ET[] = { "A", "A"};
    ExamplesTrain ex;
    BOOST_CHECK_EQUAL( ex.getMajorCategory(), AttrDomain::getUnknownId() );
    ex.push_back( createExample( ET, ET + 2, "0", classifier) );
    BOOST_CHECK_EQUAL( ex.getMajorCategory(), classifier.getCategoryIdd("0") );
    ex.push_back( createExample( ET, ET + 2, "1", classifier) );
    ex.push_back( createExample( ET, ET + 2, "1", classifier) );
    BOOST_CHECK_EQUAL( ex.getMajorCategory(), classifier.getCategoryIdd("1") );
}

BOOST_AUTO_TEST_CASE( exampleEntropy ) {

    typedef Classifier<ValueNominal<string> > Classifier;
    typedef Classifier::AttrDomain AttrDomain;
    typedef Classifier::Domains Domains;
    typedef Classifier::ExamplesTrain ExamplesTrain;

    AttrDomain cat("c");
    cat.insert("0");
    cat.insert("1");
    cat.insert("2");

    Domains domains;
    string A[] = {"A", "B", "C" };
    domains.push_back( createDomain("x", A, A + 3) );
    domains.push_back( createDomain("y", A, A + 3) );

    DummyCountClassifier classifier(domains, cat);

    string ET[] = { "A", "A"};
    ExamplesTrain ex;
    BOOST_CHECK_SMALL( ex.entropy(), 0.01 );
    ex.push_back( createExample( ET, ET + 2, "0", classifier) );
    BOOST_CHECK_SMALL( ex.entropy(), 0.01 );
    ex.push_back( createExample( ET, ET + 2, "0", classifier) );
    BOOST_CHECK_SMALL( ex.entropy(), 0.01 );
    double entr = ex.entropy();
    ex.push_back( createExample( ET, ET + 2, "1", classifier) );
    BOOST_CHECK( ex.entropy() >  entr); //entropy increases
    entr = ex.entropy();
    ex.push_back( createExample( ET, ET + 2, "1", classifier) );
    BOOST_CHECK( ex.entropy() >  entr); //entropy increases
    entr = ex.entropy();
    ex.push_back( createExample( ET, ET + 2, "2", classifier) );
    BOOST_CHECK( ex.entropy() >  entr); //entropy increases
    entr = ex.entropy();
    ex.push_back( createExample( ET, ET + 2, "2", classifier) );
    BOOST_CHECK( ex.entropy() >  entr); //entropy increases
}



namespace {

    string ATTRIBS[2] = { "A", "B" };
    const int EXAMPLE_SIZE = 4;
    string EXAMPLES[EXAMPLE_SIZE][2] = { { "A", "A" },
                                         { "A", "B" },
                                         { "B", "A" },
                                         { "B", "B" } };

} //namespace

BOOST_AUTO_TEST_CASE( classifierInterfaceTest ) {

    typedef ValueNominal<string> Value;
    typedef Classifier<Value> Classifier;
    typedef Classifier::AttrDomain AttrDomain;
    typedef Classifier::Domains Domains;
    typedef Classifier::ExampleTrain ExampleTrain;
    typedef Classifier::ExamplesTrain ExamplesTrain;

    AttrDomain a = createDomain("x", ATTRIBS, ATTRIBS + 2);
    Domains d; d.push_back(a); d.push_back(a);
    DummyCountClassifier classifier(d,a);

    ExampleTrain exx = createExample(EXAMPLES[0],EXAMPLES[0]+2,"A", classifier);
    stringstream ss;
    ss << exx;
    BOOST_CHECK( ss.str() == string("x=A x=A : x=A") );
    ExamplesTrain ex;
    ex.push_back( createExample(EXAMPLES[0],EXAMPLES[0]+2,"A", classifier ) );
    ex.push_back( createExample(EXAMPLES[1],EXAMPLES[1]+2,"A", classifier ) );
    ex.push_back( createExample(EXAMPLES[2],EXAMPLES[2]+2,"A", classifier ) );
    ex.push_back( createExample(EXAMPLES[3],EXAMPLES[3]+2,"B", classifier) );

    BOOST_CHECK( classifier.getCount() == 0 );
    classifier.train(ex);
    BOOST_CHECK( classifier.getCount() == static_cast<int>(ex.size()) );
    classifier.reset();
    BOOST_CHECK( classifier.getCount() == 0 );
}

BOOST_AUTO_TEST_CASE( classifierOstreamTest ) {

    typedef ValueNominal<string> Value;
    typedef Classifier<Value> Classifier;
    typedef Classifier::AttrDomain AttrDomain;
    typedef Classifier::Domains Domains;

    AttrDomain a = createDomain("x", ATTRIBS, ATTRIBS + 2);
    Domains d; d.push_back(a); d.push_back(a);
    DummyCountClassifier classifier(d,a);
	BOOST_CHECK_EQUAL( boost::lexical_cast<string>(classifier), string("Categories(2)x=A,x=B,;\nAttributes(2):x=A,x=B,;x=A,x=B,;") );
	//cout << classifier << endl;
}

BOOST_AUTO_TEST_CASE( classifierLoadStoreTest ) {

    typedef Classifier<ValueNominal<string> > Classifier;
    typedef Classifier::AttrDomain AttrDomain;
    typedef Classifier::Domains Domains;

    AttrDomain a = createDomain("x", ATTRIBS, ATTRIBS + 2);
	Domains d; d.push_back(a); d.push_back(a);
    DummyCountClassifier cl(d,a);

	ostringstream oss;
	boost::archive::text_oarchive oa(oss);
	oa << cl;

	//cout << oss.str() << endl;

	DummyCountClassifier classifier; //empty classifier
	BOOST_CHECK( classifier.getAttrDomains().size() == 0 );
	BOOST_CHECK( classifier.getCategoryDomain().getSize() == 0 );

    std::istringstream iss(oss.str());
    boost::archive::text_iarchive ia(iss);

	ia >> classifier;

	BOOST_CHECK( classifier.getAttrDomains().size() == 2 );
	BOOST_CHECK( classifier.getCategoryDomain().getSize() == 2 );
}

BOOST_AUTO_TEST_CASE( classifierLoadStoreXmlTest ) {

    typedef Classifier<ValueNominal<string> > Classifier;
    typedef Classifier::AttrDomain AttrDomain;
    typedef Classifier::Domains Domains;

    AttrDomain a = createDomain("x", ATTRIBS, ATTRIBS + 2);
	Domains d; d.push_back(a); d.push_back(a);
    DummyCountClassifier cl(d,a);

	ostringstream oss;
	boost::archive::xml_oarchive oa(oss);
	oa << boost::serialization::make_nvp("Classifier", cl );

	//cout << oss.str() << endl;

	DummyCountClassifier classifier; //empty classifier
	BOOST_CHECK( classifier.getAttrDomains().size() == 0 );
	BOOST_CHECK( classifier.getCategoryDomain().getSize() == 0 );

    std::istringstream iss(oss.str());
    boost::archive::xml_iarchive ia(iss);

	ia >> boost::serialization::make_nvp("Classifier", classifier);

	BOOST_CHECK( classifier.getAttrDomains().size() == 2 );
	BOOST_CHECK( classifier.getCategoryDomain().getSize() == 2 );
}

namespace {

	typedef boost::variate_generator<boost::mt19937&, boost::uniform_int<> > BoostRandom;

    /** pomocniczy funktor */
    struct ShuffleFunctor {
        BoostRandom& r_;
        ShuffleFunctor(BoostRandom& r)
            : r_(r) {
        }
        int operator()(int){ return r_(); }
    private:
        ShuffleFunctor& operator=(const ShuffleFunctor&); //forbidden
    };
}

BOOST_AUTO_TEST_CASE( stdRandomShuffleTest ) {

    int TAB[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const int TAB_SIZE = sizeof(TAB)/sizeof(TAB[0]);

    boost::mt19937 rng;
    rng.seed( (boost::uint32_t)time( 0L ) );
    boost::uniform_int<> mapping(0, TAB_SIZE - 1);
    BoostRandom gen( rng, mapping );
    ShuffleFunctor shuffleFunctor( gen );

    int tab[TAB_SIZE];
    copy(TAB, TAB + TAB_SIZE, tab );
    BOOST_CHECK( equal( TAB, TAB + TAB_SIZE, tab ) );
    random_shuffle( tab, tab + TAB_SIZE, shuffleFunctor );
    BOOST_CHECK( ! equal( TAB, TAB + TAB_SIZE, tab ) );

    int tab2[TAB_SIZE];
    copy(TAB, TAB + TAB_SIZE, tab2 );
    BOOST_CHECK( equal( TAB, TAB + TAB_SIZE, tab2 ) );
    random_shuffle( tab2, tab2 + TAB_SIZE, shuffleFunctor );
    BOOST_CHECK( ! equal( TAB, TAB + TAB_SIZE, tab2 ) );
    BOOST_CHECK( ! equal( tab, tab + TAB_SIZE, tab2 ) );
}


BOOST_AUTO_TEST_CASE( stdCreatePointerTest ) {

    int TAB[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const int TAB_SIZE = sizeof(TAB)/sizeof(TAB[0]);

    int* tab[TAB_SIZE];
    transform( TAB, TAB + TAB_SIZE, tab, & boost::lambda::_1 );

    stringstream ss;
    std::transform(tab, tab + TAB_SIZE, std::ostream_iterator<int>(ss," "), * boost::lambda::_1 );
    BOOST_CHECK( ss.str() == "1 2 3 4 5 6 7 8 9 " );
}


namespace {

    /** Klasyfikator do testow walidatora.
        Przy testach dostarcza kategorii, ktora byla kategoria ostatnio dodanego przykladu trenujacego.
    */
    class DummyClassifier : public Classifier<ValueNominal<string> > {
    public:
        DummyClassifier(const Domains& attr, const AttrDomain& category)
            : Classifier(attr, category), cat_(0L) {
        }

        virtual ~DummyClassifier() {}

        virtual void reset(){ cat_ = 0L; }

        /** metoda dodaje przyklad trenujacy (uczy) */
        virtual void train(const ExamplesTrain& c) { cat_ = c.back().getFeature(); }

        /** metoda klasyfikuje podany przyklad */
        virtual AttrIdd getCategory(const ExampleTest&) const { return cat_; }

        /** implementuje, bo musi (metoda abstrakcyjna u rodzica */
        virtual Beliefs getCategories(const ExampleTest& example) const { return Beliefs(); }

    private:
        AttrIdd cat_;
    };

} //namespace


BOOST_AUTO_TEST_CASE( classifierDummyTest ) {

    typedef Classifier<ValueNominal<string> > Classifier;
    typedef Classifier::AttrDomain AttrDomain;
    typedef Classifier::Domains Domains;
    typedef Classifier::ExamplesTrain ExamplesTrain;

    AttrDomain d("x"); d.insert("A"); d.insert("B");
    Domains dd; dd.push_back(d); dd.push_back(d);
    DummyClassifier classifier(dd,d);

    ExamplesTrain ex1;
    ex1.push_back( createExample(EXAMPLES[0],EXAMPLES[0]+2,"A", classifier ) );
    ex1.push_back( createExample(EXAMPLES[1],EXAMPLES[1]+2,"A", classifier ) );
    ex1.push_back( createExample(EXAMPLES[2],EXAMPLES[2]+2,"A", classifier ) );
    ex1.push_back( createExample(EXAMPLES[3],EXAMPLES[3]+2,"B", classifier ) );

    ExamplesTrain ex2;
    ex2.push_back( ex1.front() );

    classifier.reset();
    classifier.train(ex1);
    BOOST_CHECK( checkClassifier(ex1, classifier ) == 1 );
    BOOST_CHECK( checkClassifier(ex2, classifier ) == 0 );

    classifier.reset();
    classifier.train(ex2);
    BOOST_CHECK( checkClassifier(ex1, classifier ) == 3 );
    BOOST_CHECK( checkClassifier(ex2, classifier ) == 1 );
}

BOOST_AUTO_TEST_CASE( classifierCrossTest ) {

    typedef Classifier<ValueNominal<string> > Classifier;
    typedef Classifier::AttrDomain AttrDomain;
    typedef Classifier::Domains Domains;
    typedef Classifier::ExamplesTrain ExamplesTrain;

    AttrDomain d("x"); d.insert("A"); d.insert("B");
    Domains dd; dd.push_back(d); dd.push_back(d);
    DummyClassifier classifier(dd,d);

    ExamplesTrain ex1;
    ex1.push_back( createExample(EXAMPLES[0],EXAMPLES[0]+2,"A", classifier ) );
    ex1.push_back( createExample(EXAMPLES[1],EXAMPLES[1]+2,"A", classifier ) );
    ex1.push_back( createExample(EXAMPLES[2],EXAMPLES[2]+2,"A", classifier ) );
    ex1.push_back( createExample(EXAMPLES[3],EXAMPLES[3]+2,"B", classifier ) );

    ExamplesTrain ex2;
    ex2.push_back( ex1.front() );

    const int NUM = 100;

    //dla rozwazanego przypadku crossValidator powinien zwracac 0.5 lub 0.75 z tym samym prawdopodobienstwem.
    const int AVG = NUM / 2;
    const int D = (int)sqrt( NUM / 4.0 );

    int val05 = 0;
    int val075 = 0;

    for(int i = 0; i < NUM; i++)
        if(checkCross(ex1, 2, classifier ) > 0.6 )
            val075++;
        else
            val05++;
    BOOST_CHECK( val075 >= AVG - 5*D && val075 <= AVG + 5*D );
    BOOST_CHECK( val05 >= AVG - 5*D && val05 < AVG + 5*D );
}

BOOST_AUTO_TEST_CASE( fusionTest ) {

    typedef ValueNominal<string>::DomainType AttrDomain;
    typedef AttrDomain::ValueId AttrIdd;
    typedef Belief<ValueNominal<string> > Bel;
    typedef Bel::Beliefs Beliefs;

    AttrDomain a("x");
    AttrIdd v1 = a.insert("A");
    AttrIdd v2 = a.insert("B");
    AttrIdd v3 = a.insert("C");

    Beliefs cwb;
    cwb.push_back( Bel(v1,0.3) );
    cwb.push_back( Bel(v2,0.3) );
    cwb.push_back( Bel(v3,0.4) );

    std::vector<Beliefs> input;
    input.push_back(cwb);

    Beliefs out = fusion<Bel>(input);

    BOOST_REQUIRE_EQUAL( out.size(), cwb.size() );
    Beliefs::const_iterator iiv1 = find_if(out.begin(), out.end(),
                                           boost::bind(equal_to<AttrIdd>(), boost::bind(&Bel::getValue, _1), v1) );
    Beliefs::const_iterator iiv2 = find_if(out.begin(), out.end(),
                                           boost::bind(equal_to<AttrIdd>(), boost::bind(&Bel::getValue, _1), v2) );
    Beliefs::const_iterator iiv3 = find_if(out.begin(), out.end(),
                                           boost::bind(equal_to<AttrIdd>(), boost::bind(&Bel::getValue, _1), v3) );
    BOOST_CHECK(iiv1 != out.end() );
    BOOST_CHECK(iiv2 != out.end() );
    BOOST_CHECK(iiv3 != out.end() );
    BOOST_CHECK_CLOSE(iiv1->getProbability(), 0.3, 0.1);
    BOOST_CHECK_CLOSE(iiv2->getProbability(), 0.3, 0.1);
    BOOST_CHECK_CLOSE(iiv3->getProbability(), 0.4, 0.1);

    // fusion of 4 similar categories with belief
    input.push_back(cwb);
    input.push_back(cwb);
    input.push_back(cwb);

    Beliefs out2 = fusion<Bel>(input);
    BOOST_REQUIRE_EQUAL( out2.size(), cwb.size() );
    Beliefs::const_iterator iv1 = find_if(out2.begin(), out2.end(),
                                          boost::bind(equal_to<AttrIdd>(), boost::bind(&Bel::getValue, _1), v1) );
    Beliefs::const_iterator iv2 = find_if(out2.begin(), out2.end(),
                                          boost::bind(equal_to<AttrIdd>(), boost::bind(&Bel::getValue, _1), v2) );
    Beliefs::const_iterator iv3 = find_if(out2.begin(), out2.end(),
                                          boost::bind(equal_to<AttrIdd>(), boost::bind(&Bel::getValue, _1), v3) );
    BOOST_CHECK(iv1 != out2.end() );
    BOOST_CHECK(iv2 != out2.end() );
    BOOST_CHECK(iv3 != out2.end() );
    BOOST_CHECK_CLOSE(iv1->getProbability(), 0.19378, 0.1);
    BOOST_CHECK_CLOSE(iv2->getProbability(), 0.19378, 0.1);
    BOOST_CHECK_CLOSE(iv3->getProbability(), 0.61244, 0.1);
}

BOOST_AUTO_TEST_CASE( fusionTest2 ) {

    typedef ValueNominal<string>::DomainType AttrDomain;
    typedef AttrDomain::ValueId AttrIdd;
    typedef Belief<ValueNominal<string> > Bel;
    typedef Bel::Beliefs Beliefs;


    AttrDomain a("x");
    AttrIdd v1 = a.insert("A");
    AttrIdd v2 = a.insert("B");
    AttrIdd v3 = a.insert("C");

    Beliefs cwb;
    cwb.push_back( Bel(v1,0.5) );
    cwb.push_back( Bel(v2,0.5) );
    cwb.push_back( Bel(v3,0.0) );

    std::vector<Beliefs> input;
    input.push_back(cwb);

    Beliefs out = fusion<Bel>(input);

    BOOST_REQUIRE_EQUAL( out.size(), cwb.size() );
    Beliefs::const_iterator iv1 = find_if(out.begin(), out.end(),
                                          boost::bind(equal_to<AttrIdd>(), boost::bind(&Bel::getValue, _1), v1) );
    Beliefs::const_iterator iv2 = find_if(out.begin(), out.end(),
                                          boost::bind(equal_to<AttrIdd>(), boost::bind(&Bel::getValue, _1), v2) );
    Beliefs::const_iterator iv3 = find_if(out.begin(), out.end(),
                                          boost::bind(equal_to<AttrIdd>(), boost::bind(&Bel::getValue, _1), v3) );
    BOOST_CHECK(iv1 != out.end() );
    BOOST_CHECK(iv2 != out.end() );
    BOOST_CHECK(iv3 != out.end() );
    BOOST_CHECK_CLOSE(iv1->getProbability(), 0.5, 0.1);
    BOOST_CHECK_CLOSE(iv2->getProbability(), 0.5, 0.1);
    BOOST_CHECK_CLOSE(iv3->getProbability(), 0.0, 0.1);

    Beliefs cwb2;
    cwb2.push_back( Bel(v1,0.5) );
    cwb2.push_back( Bel(v3,0.5) );

    input.push_back(cwb2);

    Beliefs out2 = fusion<Bel>(input);
    BOOST_REQUIRE_EQUAL( out2.size(), cwb.size() );
    Beliefs::const_iterator iiv1 = find_if(out2.begin(), out2.end(),
                                           boost::bind(equal_to<AttrIdd>(), boost::bind(&Bel::getValue, _1), v1) );
    Beliefs::const_iterator iiv2 = find_if(out2.begin(), out2.end(),
                                           boost::bind(equal_to<AttrIdd>(), boost::bind(&Bel::getValue, _1), v2) );
    Beliefs::const_iterator iiv3 = find_if(out2.begin(), out2.end(),
                                           boost::bind(equal_to<AttrIdd>(), boost::bind(&Bel::getValue, _1), v3) );
    BOOST_CHECK(iiv1 != out2.end() );
    BOOST_CHECK(iiv2 != out2.end() );
    BOOST_CHECK(iiv3 != out2.end() );
    BOOST_CHECK_CLOSE( iiv1->getProbability(), 1.0, 0.1 );
    BOOST_CHECK_CLOSE( iiv2->getProbability(), 0.0, 0.1 );
    BOOST_CHECK_CLOSE( iiv3->getProbability(), 0.0, 0.1 );

}

BOOST_AUTO_TEST_CASE( fusionTestInt ) {

    typedef ValueNominal<int>::DomainType AttrDomain;
    typedef AttrDomain::ValueId AttrIdd;
    typedef Belief<ValueNominal<int> > Bel;
    typedef Bel::Beliefs Beliefs;


    AttrDomain a("x");
    AttrIdd v1 = a.insert(1);
    AttrIdd v2 = a.insert(2);
    AttrIdd v3 = a.insert(3);

    Beliefs cwb;
    cwb.push_back( Bel(v1,0.5) );
    cwb.push_back( Bel(v2,0.5) );
    cwb.push_back( Bel(v3,0.0) );

    std::vector<Beliefs> input;
    input.push_back(cwb);

    Beliefs out = fusion<Bel>(input);

    BOOST_REQUIRE_EQUAL( out.size(), cwb.size() );
    Beliefs::const_iterator iv1 = find_if(out.begin(), out.end(),
                                          boost::bind(equal_to<AttrIdd>(), boost::bind(&Bel::getValue, _1), v1) );
    Beliefs::const_iterator iv2 = find_if(out.begin(), out.end(),
                                          boost::bind(equal_to<AttrIdd>(), boost::bind(&Bel::getValue, _1), v2) );
    Beliefs::const_iterator iv3 = find_if(out.begin(), out.end(),
                                          boost::bind(equal_to<AttrIdd>(), boost::bind(&Bel::getValue, _1), v3) );
    BOOST_CHECK(iv1 != out.end() );
    BOOST_CHECK(iv2 != out.end() );
    BOOST_CHECK(iv3 != out.end() );
    BOOST_CHECK_CLOSE(iv1->getProbability(), 0.5, 0.1);
    BOOST_CHECK_CLOSE(iv2->getProbability(), 0.5, 0.1);
    BOOST_CHECK_CLOSE(iv3->getProbability(), 0.0, 0.1);

    Beliefs cwb2;
    cwb2.push_back( Bel(v1,0.5) );
    cwb2.push_back( Bel(v3,0.5) );

    input.push_back(cwb2);

    Beliefs out2 = fusion<Bel>(input);
    BOOST_REQUIRE_EQUAL( out2.size(), cwb.size() );
    Beliefs::const_iterator iiv1 = find_if(out2.begin(), out2.end(),
                                           boost::bind(equal_to<AttrIdd>(), boost::bind(&Bel::getValue, _1), v1) );
    Beliefs::const_iterator iiv2 = find_if(out2.begin(), out2.end(),
                                           boost::bind(equal_to<AttrIdd>(), boost::bind(&Bel::getValue, _1), v2) );
    Beliefs::const_iterator iiv3 = find_if(out2.begin(), out2.end(),
                                           boost::bind(equal_to<AttrIdd>(), boost::bind(&Bel::getValue, _1), v3) );
    BOOST_CHECK(iiv1 != out2.end() );
    BOOST_CHECK(iiv2 != out2.end() );
    BOOST_CHECK(iiv3 != out2.end() );
    BOOST_CHECK_CLOSE( iiv1->getProbability(), 1.0, 0.1 );
    BOOST_CHECK_CLOSE( iiv2->getProbability(), 0.0, 0.1 );
    BOOST_CHECK_CLOSE( iiv3->getProbability(), 0.0, 0.1 );
}


BOOST_AUTO_TEST_CASE( fusionTestUnknown ) {

    typedef ValueNominal<string>::DomainType AttrDomain;
    typedef AttrDomain::ValueId AttrIdd;
    typedef Belief<ValueNominal<string> > Bel;
    typedef Bel::Beliefs Beliefs;


    AttrDomain a("x");
    AttrIdd v1 = a.insert("A");
    AttrIdd v2 = a.insert("B");
    AttrIdd v3 = a.insert("C");
    AttrIdd v4 = a.insert("D");

    Beliefs cwb;
    cwb.push_back( Bel(v1,0.5) );
    cwb.push_back( Bel(v2,0.4) );
    cwb.push_back( Bel(v3,0.1) );

    std::vector<Beliefs> input;
    input.push_back(cwb);

    Beliefs cwb2;
    cwb2.push_back( Bel(v1,0.5) );
    cwb2.push_back( Bel(v3,0.1) );
    cwb2.push_back( Bel(v4,0.4) );

    input.push_back(cwb2);

    Beliefs out2 = fusion<Bel>(input);
    BOOST_REQUIRE_EQUAL( out2.size(), cwb.size() );
    Beliefs::const_iterator iiv1 = find_if(out2.begin(), out2.end(),
                                           boost::bind(equal_to<AttrIdd>(), boost::bind(&Bel::getValue, _1), v1) );
    Beliefs::const_iterator iiv2 = find_if(out2.begin(), out2.end(),
                                           boost::bind(equal_to<AttrIdd>(), boost::bind(&Bel::getValue, _1), v2) );
    Beliefs::const_iterator iiv3 = find_if(out2.begin(), out2.end(),
                                           boost::bind(equal_to<AttrIdd>(), boost::bind(&Bel::getValue, _1), v3) );
    BOOST_CHECK(iiv1 != out2.end() );
    BOOST_CHECK(iiv2 != out2.end() );
    BOOST_CHECK(iiv3 != out2.end() );
    BOOST_CHECK_CLOSE( iiv1->getProbability(), 0.25/(0.25+0.16+0.01), 0.01 );
    BOOST_CHECK_CLOSE( iiv2->getProbability(), 0.16/(0.25+0.16+0.01), 0.01 );
    BOOST_CHECK_CLOSE( iiv3->getProbability(), 0.01/(0.25+0.16+0.01), 0.01 );

}

BOOST_AUTO_TEST_SUITE_END()
