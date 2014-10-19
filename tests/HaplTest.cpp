/** plik zawiera test klas - prymitywow dla biblioteki faif */

#if defined(_MSC_VER) && (_MSC_VER >= 1400)
//msvc9.0 generuje smieci dla boost/date_time
#pragma warning(disable:4244)
#endif

#include <iostream>
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "../src/hapl/Loci.hpp"

using namespace std;
using namespace boost;
using boost::unit_test::test_suite;

using namespace faif;
using namespace faif::hapl;

BOOST_AUTO_TEST_SUITE( FAIF_haplotype_primitives_test )

BOOST_AUTO_TEST_CASE( AlleleTest ) {
    //test c-tor (silent allele)
    Allele a( AlleleImpl("a", true), 0L);
    BOOST_CHECK( getName(a) == string("a") );
    BOOST_CHECK( isSilent(a) );

    //test copy c-tor
    Allele b(a);
    BOOST_CHECK( getName(b) == string("a") );
    BOOST_CHECK( isSilent(b) );

    //test c-tor (not silent allele)
    Allele c( AlleleImpl("c",false), 0L);
    BOOST_CHECK( getName(c) == string("c") );
    BOOST_CHECK( ! isSilent(c) );

    //test assignment operators
    BOOST_CHECK ( a == b );
    BOOST_CHECK ( ! (a == c) );
}

BOOST_AUTO_TEST_CASE( LocusTest ) {
    //tworzenie obiektu, locus zawiera niemy allel
    Locus l("A");
	l.insert( AlleleImpl("A0", true) );
	l.insert( AlleleImpl("A1", false) );

    BOOST_CHECK( l.getId() == string("A") );
    BOOST_CHECK( l.getSize() == 2 );
    Locus::const_iterator it_lvar = l.begin();
    const Allele& lv1 = *(it_lvar++);
    const Allele& lv2 = *(it_lvar++);
    //dodatkowa kontrola, ze size == 2
    BOOST_CHECK( it_lvar == l.end() );

    //sprawdza warianty
    BOOST_CHECK( getName(lv1) == string("A0") );
    BOOST_CHECK( isSilent(lv1) );
    BOOST_CHECK( getName(lv2) == string("A1") );
    BOOST_CHECK( ! isSilent(lv2) );

    //test konstruktora kopiujacego
    Locus m(l);
    BOOST_CHECK( m.getSize() == 2 );

    //test tworenie obiektu, locus nie zawiera niemych alleli
    Locus n = createLocus("B", 3, false);
    BOOST_CHECK( n.getSize() == 3 );

    Locus::const_iterator it_nvar = n.begin();
    const Allele& nv1 = *(it_nvar++);
    const Allele& nv2 = *(it_nvar++);
    const Allele& nv3 = *(it_nvar++);
    BOOST_CHECK( it_nvar == n.end() );

    BOOST_CHECK( getName(nv1) == string("B1") );
    BOOST_CHECK( getName(nv2) == string("B2") );
    BOOST_CHECK( getName(nv3) == string("B3") );

    //testowanie operatora rownosci
    BOOST_CHECK( l == m );
    BOOST_CHECK( ! (l == n) );

    //testowania znajdowania odp. wariantow
    BOOST_CHECK( *n.find(AlleleImpl("B1")) == nv1 );
    BOOST_CHECK( *n.find(AlleleImpl("B2")) == nv2 );
    BOOST_CHECK( *n.find(AlleleImpl("B3")) == nv3 );

    Locus k = createLocus("C",2, true);
    k = n;
    BOOST_CHECK( *k.find(AlleleImpl("B1")) != nv1  );
    BOOST_CHECK( *k.find(AlleleImpl("B2")) != nv2  );
    BOOST_CHECK( *k.find(AlleleImpl("B3")) != nv3  );
}

BOOST_AUTO_TEST_CASE( LociTest ) {
    Loci loci;
    loci.push_back( createLocus("A", 2, true ) );
    loci.push_back( createLocus("B", 2, false ) );

	BOOST_CHECK( loci.size() == 2U );
    //  cout << loci << endl;

}

BOOST_AUTO_TEST_SUITE_END()
