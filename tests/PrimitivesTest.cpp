// plik zawiera test klas - prymitywow dla biblioteki faif

#if defined(_MSC_VER) && (_MSC_VER >= 1400)
//msvc9.0 generuje smieci dla boost/date_time
#pragma warning(disable:4244)
//OK warning for boost::serializable (BOOST_CLASS_TRACKING), but not time now to resolve it correctly
#pragma warning(disable:4308)
#endif

#include <iostream>
#include <sstream>

#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/array.hpp>

#include <boost/serialization/nvp.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>


#include "../src/Version.hpp"
#include "../src/ExceptionsFaif.hpp"
#include "../src/Value.hpp"
#include "../src/Point.hpp"

using namespace std;
using namespace boost;
using boost::unit_test::test_suite;

using namespace faif;

BOOST_AUTO_TEST_SUITE( faif_primitives_test )

BOOST_AUTO_TEST_CASE( TestVersion )
{
//call the library version functions
	BOOST_CHECK( FAIF_VERSION_MAJOR >= 0 );

	BOOST_CHECK( FAIF_VERSION_MINOR >= 0 );

	BOOST_CHECK( FAIF_VERSION_COMPILATION > 0 );
	cout << "FAIF, ver " << FAIF_VERSION_MAJOR << '.' << FAIF_VERSION_MINOR << '.' << FAIF_VERSION_COMPILATION << endl;
}

BOOST_AUTO_TEST_CASE(TestValueUnknown) {
	ValueNominal<int> v1;
	BOOST_CHECK( v1.isUnknown() );

	BOOST_CHECK(ValueNominal<int>::getUnknown().isUnknown() );

	ValueNominal<int>::DomainType d1("a");
	const ValueNominal<int>& v2 = *(d1.insert(1));
	BOOST_CHECK( ! v2.isUnknown() );
	BOOST_CHECK( ValueNominal<int>::DomainType::getUnknownId()->isUnknown() );

	ValueNominal<string> v3;
	BOOST_CHECK( v3.isUnknown() );

	ValueNominal<string>::DomainType d2("a");
	const ValueNominal<string>& v4 = *d2.insert("a");
	BOOST_CHECK( ! v4.isUnknown() );

	BOOST_CHECK( boost::lexical_cast<string>(ValueNominal<int>::getUnknown()) == string("Unknown") );
}

BOOST_AUTO_TEST_CASE(TestAttribInt)
{

	typedef ValueNominal<int> Val;
	typedef Val::DomainType Domain;

	/** without domain */
	Val x(1, 0L), y(2, 0L);

	BOOST_CHECK_EQUAL( x.get(), 1 );
	BOOST_CHECK_EQUAL( y.get(), 2 );

	BOOST_CHECK( x != y );
	BOOST_CHECK( x == Val(1, 0L) );

	Domain d1("a");
	const Val& a1 = *d1.insert( 1 );
	const Val& a2 = *d1.insert( 2 );
	BOOST_CHECK_EQUAL( d1.getSize(), 2 );

	BOOST_CHECK( a1 != a2 );
	BOOST_CHECK( a2 == *d1.find( 2 ) );

	BOOST_CHECK_THROW( d1.find(555), NotFoundException );

	Val::DomainType d2(d1);
	BOOST_CHECK_EQUAL( d2.getSize(), 2 );

 	const Val& a3 = *d2.find( 1 );
 	BOOST_CHECK( a1 != a3 );
    BOOST_CHECK_EQUAL( lexical_cast<string>(a1), string("a=1") );
    BOOST_CHECK_EQUAL( lexical_cast<string>(d2.find(1)), string("a=1") );
	BOOST_CHECK_EQUAL( lexical_cast<string>(d2), string("a=1,a=2,;") );
}

BOOST_AUTO_TEST_CASE( TestAttribString )
{
	typedef ValueNominal<string> Val;
	typedef Val::DomainType Domain;

	Domain d1("a");
	const Val& a1 = *d1.insert( "x" );
	d1.insert( "y" );

	Domain d2("b");
	const Val& a3 = *d2.insert( "z" );

	BOOST_CHECK( ! (a1 == a3) );
	BOOST_CHECK( a1 != a3 );

	d2 = d1;
	BOOST_CHECK_EQUAL( d2.getSize(), 2 );

	BOOST_CHECK_EQUAL( lexical_cast<string>(d2), string("a=x,a=y,;") );

	BOOST_CHECK_NO_THROW( d2.find("x"));
    BOOST_CHECK_EQUAL( lexical_cast<string>(d2.find("x")), string("a=x") );
	BOOST_CHECK_THROW( d2.find("xx"), NotFoundException );
}

BOOST_AUTO_TEST_CASE( TestCreateDomain )
{
	string A1[] = {"x", "y", "z"};
	const int SIZE1 = sizeof(A1)/sizeof(A1[0]);
	ValueNominal<string>::DomainType d1 = createDomain("a", A1, A1 + SIZE1 );
	BOOST_CHECK_EQUAL( d1.getSize(), 3 );
	BOOST_CHECK_EQUAL( lexical_cast<string>(d1), string("a=x,a=y,a=z,;") );

	int A[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	const int SIZE = sizeof(A)/sizeof(A[0]);
	ValueNominal<int>::DomainType d2 = createDomain<int>("b", A, A + SIZE );
	BOOST_CHECK_EQUAL( d2.getSize(), SIZE );
	BOOST_CHECK_EQUAL( lexical_cast<string>(d2), string("b=1,b=2,b=3,b=4,b=5,b=6,b=7,b=8,b=9,b=10,;") );
}

BOOST_AUTO_TEST_CASE( TestSerializeDomainStr )
{
	string A1[] = {"x", "y", "z"};
	const int SIZE1 = sizeof(A1)/sizeof(A1[0]);
	ValueNominal<string>::DomainType d1 = createDomain("a", A1, A1 + SIZE1 );

	ostringstream oss;
	boost::archive::text_oarchive oa(oss);
	oa << d1;
	//cout << oss.str() << endl;

	ValueNominal<string>::DomainType d2;

    std::istringstream iss(oss.str());
    boost::archive::text_iarchive ia(iss);

	ia >> d2;
	BOOST_CHECK_EQUAL( d1.getSize(), d2.getSize() );
	BOOST_CHECK_EQUAL( d2.getId(), "a" );
	BOOST_CHECK_NO_THROW( d2.find("x") );
	BOOST_CHECK_NO_THROW( d2.find("y") );
	BOOST_CHECK_NO_THROW( d2.find("z") );

	const ValueNominal<string>& vx = *d2.find("x");
	const ValueNominal<string>& vy = *d2.find("y");
	const ValueNominal<string>& vz = *d2.find("z");

	BOOST_CHECK_EQUAL( vx.get(), "x" );
	BOOST_CHECK_EQUAL( vy.get(), "y" );
	BOOST_CHECK_EQUAL( vz.get(), "z" );

	BOOST_CHECK_EQUAL( vx.getDomain(), &d2 );
	BOOST_CHECK_EQUAL( vy.getDomain(), &d2 );
	BOOST_CHECK_EQUAL( vz.getDomain(), &d2 );
}

BOOST_AUTO_TEST_CASE( TestSerializeDomainInt )
{
	int A1[] = { 1, 2, 3};
	const int SIZE1 = sizeof(A1)/sizeof(A1[0]);
	ValueNominal<int>::DomainType d1 = createDomain("a", A1, A1 + SIZE1 );

	ostringstream oss;
	boost::archive::text_oarchive oa(oss);
	oa << d1;
	//cout << oss.str() << endl;

	ValueNominal<int>::DomainType d2;

    std::istringstream iss(oss.str());
    boost::archive::text_iarchive ia(iss);

	ia >> d2;
	BOOST_CHECK_EQUAL( d1.getSize(), d2.getSize() );
	BOOST_CHECK_EQUAL( d2.getId(), "a" );
	BOOST_CHECK_NO_THROW(d2.find(1) );
	BOOST_CHECK_NO_THROW(d2.find(2) );
	BOOST_CHECK_NO_THROW(d2.find(3) );

	const ValueNominal<int>& v1 = *d2.find(1);
	const ValueNominal<int>& v2 = *d2.find(2);
	const ValueNominal<int>& v3 = *d2.find(3);

	BOOST_CHECK_EQUAL( v1.get(), 1 );
	BOOST_CHECK_EQUAL( v2.get(), 2 );
	BOOST_CHECK_EQUAL( v3.get(), 3 );

	BOOST_CHECK_EQUAL( v1.getDomain(), &d2 );
	BOOST_CHECK_EQUAL( v2.getDomain(), &d2 );
	BOOST_CHECK_EQUAL( v3.getDomain(), &d2 );
}

BOOST_AUTO_TEST_CASE( TestSerializeDomainXML )
{
	string A1[] = {"x", "y", "z"};
	const int SIZE1 = sizeof(A1)/sizeof(A1[0]);
	ValueNominal<string>::DomainType d1 = createDomain("a", A1, A1 + SIZE1 );

	ostringstream oss;
	boost::archive::xml_oarchive oa(oss);
	oa << boost::serialization::make_nvp("Domain", d1);
	//cout << oss.str() << endl;

	ValueNominal<string>::DomainType d2;

    std::istringstream iss(oss.str());
    boost::archive::xml_iarchive ia(iss);

	ia >> boost::serialization::make_nvp("Domain", d2);
	BOOST_CHECK_EQUAL( d1.getSize(), d2.getSize() );
	BOOST_CHECK_EQUAL( d2.getId(), "a" );
	BOOST_CHECK_NO_THROW(d2.find("x") );
	BOOST_CHECK_NO_THROW(d2.find("y") );
	BOOST_CHECK_NO_THROW(d2.find("z") );

	const ValueNominal<string>& vx = *d2.find("x");
	const ValueNominal<string>& vy = *d2.find("y");
	const ValueNominal<string>& vz = *d2.find("z");

	BOOST_CHECK_EQUAL( vx.get(), "x" );
	BOOST_CHECK_EQUAL( vy.get(), "y" );
	BOOST_CHECK_EQUAL( vz.get(), "z" );

	BOOST_CHECK_EQUAL( vx.getDomain(), &d2 );
	BOOST_CHECK_EQUAL( vy.getDomain(), &d2 );
	BOOST_CHECK_EQUAL( vz.getDomain(), &d2 );
}

BOOST_AUTO_TEST_CASE( TestSpace )
{
	typedef ValueNominal<string>::DomainType DomainString;

	string A[] = {"A", "B", "C" };
	const unsigned int A_LEN = sizeof(A)/sizeof(A[0]);
	Space<DomainString> domains;
	domains.push_back(createDomain("x", A, A + A_LEN) );
	domains.push_back(createDomain("y", A, A + A_LEN) );

    string ET[] = { "A", "A"};
	const unsigned int ET_LEN = sizeof(ET)/sizeof(ET[0]);
	Point<ValueNominal<string> > et = domains.createPoint( ET, ET + ET_LEN);
	//cout << boost::lexical_cast<string>(et) << endl;
	BOOST_CHECK( boost::lexical_cast<string>(et) == string("x=A y=A ") );


	typedef ValueNominal<int>::DomainType DomainInt;

	int B[] = {1, 2, 3, 4};
	const unsigned int B_LEN = sizeof(B)/sizeof(B[0]);
	Space<DomainInt> domains2;
	domains2.push_back(createDomain("x", B, B + B_LEN) );
	domains2.push_back(createDomain("y", B, B + B_LEN) );

    int ET2[] = { 1, 4 };
	const unsigned int ET2_LEN = sizeof(ET2)/sizeof(ET2[0]);
	Point<ValueNominal<int> > et2 = domains2.createPoint( ET2, ET2 + ET2_LEN);
	//cout << boost::lexical_cast<string>(et2) << endl;
	BOOST_CHECK( boost::lexical_cast<string>(et2) == string("x=1 y=4 ") );

    int ET3[] = { 1, 44 };
	const unsigned int ET3_LEN = sizeof(ET3)/sizeof(ET3[0]);
	BOOST_CHECK_THROW( domains2.createPoint( ET3, ET3 + ET3_LEN), NotFoundException );

}

BOOST_AUTO_TEST_CASE( TestSpace2 )
{
	typedef ValueNominal<string>::DomainType DomainString;

	string A[] = {"A", "B", "C" };
	const unsigned int A_LEN = sizeof(A)/sizeof(A[0]);
	Space<DomainString> domains;
	domains.push_back(createDomain("x", A, A + A_LEN) );
	domains.push_back(createDomain("y", A, A + A_LEN) );

	//the creator from pairs <categ_name, value>
	vector<pair<string,string> > v;
	v.push_back( make_pair<string,string>("y","B") ); //the different sequence
	v.push_back( make_pair<string,string>("x","A") );

	Point<ValueNominal<string> > et = domains.createPoint(v);
	//cout << boost::lexical_cast<string>(et) << endl;
	BOOST_CHECK( boost::lexical_cast<string>(et) == string("x=A y=B ") );

	vector<pair<string,string> > v2;
	v2.push_back( make_pair<string,string>("y","YY") ); //the different sequence
	v2.push_back( make_pair<string,string>("xx","A") );

	Point<ValueNominal<string> > et2 = domains.createPoint(v2);
	BOOST_CHECK( boost::lexical_cast<string>(et2) == string("Unknown Unknown ") );

	vector<pair<string,string> > v3;
	//empty example test
	Point<ValueNominal<string> > et3 = domains.createPoint(v3);
	BOOST_CHECK( boost::lexical_cast<string>(et3) == string("Unknown Unknown ") );

	v3.push_back( make_pair<string,string>("y","B") ); //the different sequence
	//no value for x
    BOOST_CHECK_THROW( domains.createPointStrict(v3), NotFoundException );
	v3.push_back( make_pair<string,string>("x","XX") ); //the different sequence
	// bad value for x
    BOOST_CHECK_THROW( domains.createPointStrict(v3), NotFoundException );
}


template<typename Feature> struct FeatureInitTestPointAndFeature {
	static Feature init() {
		return string("Test");
	}
};


BOOST_AUTO_TEST_CASE( TestPointAndFeature ) {

	typedef ValueNominal<string>::DomainType DomainString;

	string A[] = {"A", "B", "C" };
	const unsigned int A_LEN = sizeof(A)/sizeof(A[0]);
	Space<DomainString> domains;
	domains.push_back(createDomain("x", A, A + A_LEN) );
	domains.push_back(createDomain("y", A, A + A_LEN) );

    string ET[] = { "A", "B"};
	const unsigned int ET_LEN = sizeof(ET)/sizeof(ET[0]);
	PointAndFeature<ValueNominal<string>, string> e;
	BOOST_CHECK_EQUAL( boost::lexical_cast<string>(e),string(": ") );

	PointAndFeature<ValueNominal<string>, string> e2(domains.createPoint( ET, ET + ET_LEN),"X");
	BOOST_CHECK_EQUAL( boost::lexical_cast<string>(e2),string("x=A y=B : X") );

	PointAndFeature<ValueNominal<string>, string, FeatureInitTestPointAndFeature> e3;
	BOOST_CHECK_EQUAL( boost::lexical_cast<string>(e3),string(": Test") );
}



BOOST_AUTO_TEST_SUITE_END()
