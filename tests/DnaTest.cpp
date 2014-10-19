/** plik zawiera test klas dna */

#include <iostream>
#include <sstream>
#include <string>
#include <limits>

#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "../src/dna/ExceptionsDna.h"
#include "../src/dna/Nucleotide.h"
#include "../src/dna/Chain.h"
#include "../src/dna/EnergyNucleo.h"
#include "../src/dna/SecStruct.h"
#include "../src/dna/FoldedChain.h"
#include "../src/dna/FoldedPair.h"
#include "../src/dna/Codon.h"
#include "../src/dna/CodonAminoTable.h"

using namespace std;
using namespace boost;
using namespace faif;
using namespace faif::dna;
using boost::unit_test::test_suite;

BOOST_AUTO_TEST_SUITE( faif_primitives_test )

BOOST_AUTO_TEST_CASE( NucleotideTest ) {
    Nucleotide nuc1(ADENINE);
    BOOST_CHECK(nuc1.get()==ADENINE);

    Nucleotide nuc2(THYMINE);
    BOOST_CHECK(nuc2.get()==THYMINE);

    Nucleotide nuc3(CYTOSINE);
    BOOST_CHECK(nuc3.get()==CYTOSINE);

    Nucleotide nuc4(GUANINE);
    BOOST_CHECK(nuc4.get()==GUANINE);

    nuc1=nuc4;
    BOOST_CHECK(nuc1.get()==GUANINE);

    BOOST_CHECK(nuc1==nuc4);
    BOOST_CHECK(nuc1!=nuc3);
    BOOST_CHECK(nuc1!=nuc2);

    BOOST_CHECK(nuc4.complementary()==nuc3);

    Nucleotide nuc;
    BOOST_WARN(nuc.get()==ADENINE);

    Nucleotide nuc01 = create('A');
    Nucleotide nuc02 = create('T');
    Nucleotide nuc03 = create('G');
    Nucleotide nuc04 = create('C');
    BOOST_CHECK(nuc01.get()==ADENINE);
    BOOST_CHECK(nuc02.get()==THYMINE);
    BOOST_CHECK(nuc03.get()==GUANINE);
    BOOST_CHECK(nuc04.get()==CYTOSINE);
}

BOOST_AUTO_TEST_CASE( NucleotideIteratorTest ){
    Chain x("GCTA");
    Chain::iterator it = x.begin();
    BOOST_CHECK((*it).get()==GUANINE); //it->0
    ++it;
    BOOST_CHECK((*it).get()==CYTOSINE);//it->1
    ++it;
    BOOST_CHECK((*it).get()==THYMINE);//it->2
    ++it;
    BOOST_CHECK((*it).get()==ADENINE);//it->3
    --it;
    BOOST_CHECK((*it).get()==THYMINE);//it->2
    --it;
    BOOST_CHECK((*it).get()==CYTOSINE);//it->1
    --it;
    BOOST_CHECK((*it).get()==GUANINE);//it->0

    (*it) = create('A');
    BOOST_CHECK((*it).get()==ADENINE);   //zmiana pierwszego nukleotydu na ADENINE

    Chain::iterator it2 = x.begin(); //it2->0 it->0
    BOOST_CHECK(it==it2);
    BOOST_CHECK(it>=it2);
    BOOST_CHECK(it<=it2);
    ++it2;
    BOOST_CHECK(it<it2); //it2->1 it->0
    BOOST_CHECK(it<=it2);
    --it2;
    ++it;
    BOOST_CHECK(it>it2); //it2->0 it->1
    BOOST_CHECK(it>=it2);
    ++it2;
    Chain::iterator it3 = it2;
    BOOST_CHECK((*it3).get()==CYTOSINE);    //it2->1 it->1
    it3+=2;
    BOOST_CHECK((*it3).get()==ADENINE);    //it->3
    it3-=1;
    BOOST_CHECK((*it3).get()==THYMINE);      //it->2

    Chain::iterator iter = x.begin();
    BOOST_CHECK((*iter).get()==ADENINE);        //it->0
    Chain::iterator iter2=x.end();
    --iter2;
    BOOST_CHECK((*iter2).get()==ADENINE);        //it->3



    Chain y("GCTA");
    Chain::iterator i1 = x.begin();
    Chain::iterator i2 = y.begin();
    Chain::iterator i3=x.begin();
    BOOST_CHECK(i1==i3);

    //juz zrobione
    //vc8.0 wyrzucal wyjatek gdy dwa iteratory z roznych kolekcji!
    BOOST_REQUIRE_NO_THROW( i1==i2 );
    BOOST_CHECK( !(i1==i2) );
}

BOOST_AUTO_TEST_CASE( ChainTest ) {
    Chain x("ATCG");            //x=ATCG
    Chain y;
    y += Nucleotide(ADENINE);
    y += Nucleotide(THYMINE);
    y += Nucleotide(CYTOSINE);
    y += Nucleotide(GUANINE);  //y=ATCG
    BOOST_CHECK(x==y);

    Chain yy(x.begin(), x.end() );
    BOOST_CHECK(x == yy);
    string test("CGAT");
    y=x.complementary();                    //y=CGAT
    BOOST_CHECK(x!=y);
    BOOST_CHECK(test==y.getString());
    BOOST_CHECK(4==x.getLength());
    Chain z;
    z+=x;
    z+=y;
    BOOST_CHECK(8==z.getLength());          //z=ATCGCGAT
    BOOST_CHECK(x!=z);
    Chain k("ATCGCGAT");                    //k=ATCGCGAT
    BOOST_CHECK(k==z);
    k= createSubChain(z,2,4);                   //k=CGCG
    BOOST_CHECK(4 == k.getLength() );
    //y = CGAT
    y[2] = create('C');                       //y=CGCT
    y[3] = create('G');                       //y=CGCG
    BOOST_CHECK(y==k);
    Chain n = createSubChain(x,0,4); //n=ATCG
    BOOST_CHECK(n==x);

    Chain p("ATCGCGAT");
    ostringstream os;
    os << p;
    BOOST_CHECK_EQUAL(os.str(),"ATCGCGAT");
}

BOOST_AUTO_TEST_CASE( EnergyTest ) {
    Nucleotide a(ADENINE);
    Nucleotide t(THYMINE);
    Nucleotide g(GUANINE);
    Nucleotide c(CYTOSINE);
    // default configuration
    EnergyNucleo energy = createDefaultEnergy();
    BOOST_CHECK_EQUAL(energy.getEnergy(a,t) , 2);
    BOOST_CHECK_EQUAL(energy.getEnergy(t,a) , 2);
    BOOST_CHECK_EQUAL(energy.getEnergy(g,c) , 3);
    BOOST_CHECK_EQUAL(energy.getEnergy(c,g) , 3);
    BOOST_CHECK_EQUAL(energy.getEnergy(g,t) , 1);
    BOOST_CHECK_EQUAL(energy.getEnergy(t,g) , 1);
    BOOST_CHECK_EQUAL(energy.getEnergy(a,c) , DEFAULT_NUCLEO_ENERGY);

    //default modified configuration
    EnergyNucleo energy2 = createDefaultEnergy(-1);
    BOOST_CHECK(energy2.getEnergy(a,t) == 2);
    BOOST_CHECK(energy2.getEnergy(t,a) == 2);
    BOOST_CHECK(energy2.getEnergy(g,c) == 3);
    BOOST_CHECK(energy2.getEnergy(c,g) == 3);
    BOOST_CHECK(energy2.getEnergy(a,c) == -1);
    // user configuration
    EnergyNucleo energy3(-1);
    energy3.addPair(a,t,1);
    energy3.addPair(a,c,2);
    energy3.addPair(a,g,3);
    energy3.addPair(t,c,4);
    energy3.addPair(t,g,5);
    BOOST_CHECK_EQUAL(energy3.getEnergy(a,t), 1);
    BOOST_CHECK_EQUAL(energy3.getEnergy(a,c), 2);
    BOOST_CHECK_EQUAL(energy3.getEnergy(a,g), 3);
    BOOST_CHECK_EQUAL(energy3.getEnergy(t,c), 4);
    BOOST_CHECK_EQUAL(energy3.getEnergy(t,g), 5);
    BOOST_CHECK_EQUAL(energy3.getEnergy(c,g), -1);
}

BOOST_AUTO_TEST_CASE( SecStructTest ) {

    Chain x(string("AAAAAAGGGTTTTTT"));

    Chain::const_iterator ib = x.begin();
    Chain::const_iterator ie = x.end();
    --ie;
    ConnectPair p1(ib, ie );
    ConnectPair p2( p1 );

    BOOST_CHECK( p1 == p2 );
    BOOST_CHECK( !(p1 < p2) );

    BOOST_CHECK_EQUAL( &x, &p1.first.getChain() );
    BOOST_CHECK_EQUAL( &x, &p1.second.getChain() );

    ostringstream os;
    BOOST_CHECK_NO_THROW( os << p1 );
    BOOST_CHECK_EQUAL( os.str(), "(0,14)" );

    SecStruct sec_struct;
    sec_struct.addPair( p1 );
    ++ib; --ie;
    sec_struct.addPair(ConnectPair(ib,ie) );
    ++ib; --ie;
    sec_struct.addPair(ConnectPair(ib,ie) );
    ++ib; --ie;
    sec_struct.addPair(ConnectPair(ib,ie) );
    ++ib; --ie;
    sec_struct.addPair(ConnectPair(ib,ie) );
    ++ib; --ie;
    sec_struct.addPair(ConnectPair(ib,ie) );

    BOOST_CHECK_EQUAL( sec_struct.size(), 6 );

    SecStruct sec_struct2(sec_struct);
    BOOST_CHECK( sec_struct == sec_struct2 );
    sec_struct2.append(sec_struct);
    BOOST_CHECK( sec_struct == sec_struct2 );
    BOOST_CHECK( !(sec_struct < sec_struct2) );

    ostringstream os2;
    BOOST_REQUIRE_NO_THROW( os2 << sec_struct );
    BOOST_CHECK_EQUAL( os2.str(), "(0,14)(1,13)(2,12)(3,11)(4,10)(5,9)");
    EnergyNucleo energy = createDefaultEnergy();
    BOOST_CHECK_EQUAL( sec_struct.energy(energy), 12 );
}

namespace {

    ConnectPair makeConnectPair(const Chain& fir, int idx_first, const Chain& sec, int idx_second) {
        Chain::const_iterator it_first = fir.begin();
        it_first += idx_first;
        Chain::const_iterator it_second = sec.begin();
        it_second += idx_second;
        return ConnectPair(it_first,it_second);
    }
}

BOOST_AUTO_TEST_CASE( FoldedChainOneStructTest ) {
    string str("AAAAAAGGGTTTTTT");
    EnergyNucleo energy = createDefaultEnergy();
    BOOST_REQUIRE_NO_THROW( FoldedChain f(Chain(str),energy,1000); );
    FoldedChain folded(Chain(str),energy,1000);

    BOOST_REQUIRE( folded.getStructures().size() > 0 );
    BOOST_CHECK( folded.getStructures().size() == 1 );
    const SecStruct& result_struct = *(folded.getStructures().begin() );

    const Chain& chain = folded.getChain();
    SecStruct required_struct;
    required_struct.addPair( makeConnectPair(chain, 0, chain, 14) );
    required_struct.addPair( makeConnectPair(chain, 1, chain, 13) );
    required_struct.addPair( makeConnectPair(chain, 2, chain, 12) );
    required_struct.addPair( makeConnectPair(chain, 3, chain, 11) );
    required_struct.addPair( makeConnectPair(chain, 4, chain, 10) );
    required_struct.addPair( makeConnectPair(chain, 5, chain, 9) );
    BOOST_CHECK(result_struct == required_struct);
    BOOST_CHECK_EQUAL(folded.getSecStructEnergy(),12);
}


BOOST_AUTO_TEST_CASE( FoldedChainManyStructTest ) {
    EnergyNucleo energy = createDefaultEnergy();

    string chain_str("GCAGCACCCAAAGGGAATATGGGATACGCGTA");
    BOOST_REQUIRE_NO_THROW( FoldedChain f(Chain(chain_str),energy,1000); );
    FoldedChain folded(Chain(chain_str),energy,1000);
    //  folded.printStructures(cout,3);
    //  cout << endl;
    //  folded.printStructures(cout);
    //  cout << endl;

    static const unsigned long EXPECT_RESULT_NUM = 6;

    const Chain& chain = folded.getChain();

    SecStruct expect[EXPECT_RESULT_NUM];

    expect[0].addPair(makeConnectPair(chain, 0, chain, 30));
    expect[0].addPair(makeConnectPair(chain, 1, chain, 29));
    expect[0].addPair(makeConnectPair(chain, 3, chain, 28));
    expect[0].addPair(makeConnectPair(chain, 4, chain, 27));
    expect[0].addPair(makeConnectPair(chain, 5, chain, 24));
    expect[0].addPair(makeConnectPair(chain, 6, chain, 22));
    expect[0].addPair(makeConnectPair(chain, 7, chain, 21));
    expect[0].addPair(makeConnectPair(chain, 8, chain, 20));
    expect[0].addPair(makeConnectPair(chain, 9, chain, 19));
    expect[0].addPair(makeConnectPair(chain,10, chain, 17));

    expect[1].addPair(makeConnectPair(chain, 0, chain, 30));
    expect[1].addPair(makeConnectPair(chain, 1, chain, 29));
    expect[1].addPair(makeConnectPair(chain, 3, chain, 28));
    expect[1].addPair(makeConnectPair(chain, 4, chain, 27));
    expect[1].addPair(makeConnectPair(chain, 5, chain, 24));
    expect[1].addPair(makeConnectPair(chain, 6, chain, 22));
    expect[1].addPair(makeConnectPair(chain, 7, chain, 21));
    expect[1].addPair(makeConnectPair(chain, 8, chain, 20));
    expect[1].addPair(makeConnectPair(chain, 9, chain, 19));
    expect[1].addPair(makeConnectPair(chain,11, chain, 17));

    expect[2].addPair(makeConnectPair(chain, 0, chain, 30));
    expect[2].addPair(makeConnectPair(chain, 1, chain, 29));
    expect[2].addPair(makeConnectPair(chain, 3, chain, 28));
    expect[2].addPair(makeConnectPair(chain, 4, chain, 27));
    expect[2].addPair(makeConnectPair(chain, 5, chain, 24));
    expect[2].addPair(makeConnectPair(chain, 6, chain, 22));
    expect[2].addPair(makeConnectPair(chain, 7, chain, 21));
    expect[2].addPair(makeConnectPair(chain, 8, chain, 20));
    expect[2].addPair(makeConnectPair(chain,10, chain, 19));
    expect[2].addPair(makeConnectPair(chain,11, chain, 17));

    expect[3].addPair(makeConnectPair(chain, 0, chain, 30));
    expect[3].addPair(makeConnectPair(chain, 1, chain, 29));
    expect[3].addPair(makeConnectPair(chain, 3, chain, 28));
    expect[3].addPair(makeConnectPair(chain, 4, chain, 27));
    expect[3].addPair(makeConnectPair(chain, 6, chain, 14));
    expect[3].addPair(makeConnectPair(chain, 7, chain, 13));
    expect[3].addPair(makeConnectPair(chain, 8, chain, 12));
    expect[3].addPair(makeConnectPair(chain,17, chain, 25));
    expect[3].addPair(makeConnectPair(chain,18, chain, 24));
    expect[3].addPair(makeConnectPair(chain,19, chain, 23));

    expect[4].addPair(makeConnectPair(chain, 0, chain, 30));
    expect[4].addPair(makeConnectPair(chain, 1, chain, 29));
    expect[4].addPair(makeConnectPair(chain, 3, chain, 28));
    expect[4].addPair(makeConnectPair(chain, 4, chain, 27));
    expect[4].addPair(makeConnectPair(chain, 7, chain, 13));
    expect[4].addPair(makeConnectPair(chain, 8, chain, 12));
    expect[4].addPair(makeConnectPair(chain,14, chain, 26));
    expect[4].addPair(makeConnectPair(chain,17, chain, 25));
    expect[4].addPair(makeConnectPair(chain,18, chain, 24));
    expect[4].addPair(makeConnectPair(chain,19, chain, 23));

    expect[5].addPair(makeConnectPair(chain, 0, chain, 30));
    expect[5].addPair(makeConnectPair(chain, 1, chain, 29));
    expect[5].addPair(makeConnectPair(chain, 3, chain, 28));
    expect[5].addPair(makeConnectPair(chain, 6, chain, 27));
    expect[5].addPair(makeConnectPair(chain, 7, chain, 13));
    expect[5].addPair(makeConnectPair(chain, 8, chain, 12));
    expect[5].addPair(makeConnectPair(chain,14, chain, 26));
    expect[5].addPair(makeConnectPair(chain,17, chain, 25));
    expect[5].addPair(makeConnectPair(chain,18, chain, 24));
    expect[5].addPair(makeConnectPair(chain,19, chain, 23));


    //cout << folded.findInDepth() << endl;
    //cout << folded << endl;

    SecStructures result = folded.getStructures();

    //      folded.printMatrix(cout,3);
    //      cout << endl;


    BOOST_CHECK_EQUAL( result.size(),EXPECT_RESULT_NUM );
    //sprawdza poprawnosc kolejnych struktur
    for(unsigned int i=0;i<EXPECT_RESULT_NUM;++i) {
        BOOST_CHECK( result.find(expect[i]) != result.end() );
    }
    BOOST_CHECK_EQUAL(folded.getSecStructEnergy(),25);
}

BOOST_AUTO_TEST_CASE( SecStructLimitTest ) {
    EnergyNucleo energy = createDefaultEnergy();
    Chain test_chain("AATAATAATAATAATAATAATAATAATAAT");
    FoldedChain folded(test_chain,energy,1000);
    SecStructures result = folded.getStructures();
    BOOST_CHECK_EQUAL(result.size(),1000UL);
    BOOST_CHECK_EQUAL(folded.getSecStructEnergy(), 18 );

    FoldedChain folded2(test_chain,energy,10000);
    result = folded2.getStructures();
    BOOST_CHECK_EQUAL(result.size(),2378UL);
    BOOST_CHECK_EQUAL(folded.getSecStructEnergy(), 18 );
}

void CheckEnergyAndStruct(const std::string& dna_chain, EnergyValue sec_str_energy, unsigned long struct_number) {
    EnergyNucleo energy = createDefaultEnergy();
    Chain chain(dna_chain);
    FoldedChain folded(chain,energy );
    BOOST_CHECK_EQUAL( folded.getSecStructEnergy(), sec_str_energy);
    BOOST_CHECK_EQUAL( folded.getStructures().size(), struct_number);
}


BOOST_AUTO_TEST_CASE( TestEnergyAndStruct1 ) {
    CheckEnergyAndStruct("AAAAAAGGGTTTTTT", 12, 1);
}

BOOST_AUTO_TEST_CASE( TestEnergyAndStruct2 ) {
    CheckEnergyAndStruct("AATAATAATAATAATAATAATAATAATAAT", 18, 100);
}

BOOST_AUTO_TEST_CASE( TestEnergyAndStruct3 ) {
    //ubikwityna UbiSyn.txt
    CheckEnergyAndStruct( "ATGCAAATTTTTGTTAAAACTTTAACTGGT"
                          "AAAACCATTACCTTAGAAGTTGAATCTTCA"
                          "GATACCATTGATAATGTTAAATCTAAAATT"
                          "CAAGATAAAGAAGGTATTCCTCCAGATCAA"
                          "CAACGTCTAATATTTGCAGGTAAACAGTTA"
                          "GAAGATGGTCGTACCCTGTCTGATTATAAC"
                          "ATTCAGAAAGAATCTACCTTACATCTGGTC"
                          "TTACGTCTCCGCGGTGGTGAAGAGCATTAA", 207, 100 );
}

// BOOST_AUTO_TEST_CASE( TestEnergyAndStruct4 ) {
//     // interferon beta IFNBeta.txt
//     CheckEnergyAndStruct( "ATGAGCTATAACCTGCTGGGTTTCCTGCAA"
// 						  "CGTAGCAGCAACTTCCAGAGCCAGAAACTG"
// 						  "CTGTGGCAGCTGAACGGTCGTCTGGAATAT"
// 						  "TGCCTGAAAGATCGTATGAACTTCGATATC"
// 						  "CCGGAAGAAATCAAACAGCTGCAGCAGTTC"
// 						  "CAGAAAGAAGATGCAGCACTGACCATCTAC"
// 						  "GAAATGCTGCAAAACATCTTCGCAATCTTC"
// 						  "CGTCAGGATAGCAGCAGCACCGGTTGGAAT"
// 						  "GAAACCATCGTTGAAAACCTGCTGGCGAAC"
// 						  "GTTTACCACCAGATTAACCACCTGAAAACC"
// 						  "GTTCTGGAAGAAAAACTGGAAAAAGAAGAT"
// 						  "TTCACCCGTGGTAAACTGATGAGCAGCCTG"
// 						  "CACCTGAAACGTTATTACGGTCGTATTCTG"
// 						  "CACTATCTGAAAGCTAAAGAATATAGCCAC"
// 						  "TGCGCGTGGACCATTGTTCGTGTTGAAATC"
// 						  "CTGCGTAACTTCTACTTCATCAACCGTCTG"
//                           "ACCGGTTACCTGCGTAACTAA",          452, 100 );
// }

BOOST_AUTO_TEST_CASE( FoldedPairWithoutSplitTest ) {
    const int MAX_FOLDINGS = 1000;
    EnergyNucleo energy = createDefaultEnergy();

    FoldedPair p( Chain(string("AAAAAGGG")), Chain(string("GGGTTTTT")), energy, MAX_FOLDINGS );

    //cout << p << endl;
    BOOST_REQUIRE( p.getStructures().size() > 0 );
    BOOST_CHECK( p.getStructures().size() == 1 );
    const SecStruct& result_struct = *(p.getStructures().begin() );

    const Chain& first = p.getFirstChain();
    const Chain& second = p.getSecondChain();

    SecStruct required_struct;
    required_struct.addPair( makeConnectPair(first, 0, second, 7) );
    required_struct.addPair( makeConnectPair(first, 1, second, 6) );
    required_struct.addPair( makeConnectPair(first, 2, second, 5) );
    required_struct.addPair( makeConnectPair(first, 3, second, 4) );
    required_struct.addPair( makeConnectPair(first, 4, second, 3) );

    BOOST_CHECK(result_struct == required_struct);
    BOOST_CHECK_EQUAL(p.getSecStructEnergy(),10);

}

BOOST_AUTO_TEST_CASE( FoldedPairWithSplitTest ) {
    const int MAX_FOLDINGS = 1000;
    EnergyNucleo energy = createDefaultEnergy();

    FoldedPair p( Chain(string("AAAACCC")), Chain(string("GGGTTTT")), energy, MAX_FOLDINGS );

    BOOST_REQUIRE( p.getStructures().size() > 0 );

    //  cout << p << endl;
    //  p.printMatrix(cout);
    //  p.printStructures(cout);

    BOOST_CHECK( p.getStructures().size() == 1 );
    const SecStruct& result_struct = *(p.getStructures().begin() );

    const Chain& first = p.getFirstChain();
    const Chain& second = p.getSecondChain();

    SecStruct required_struct;
    required_struct.addPair( makeConnectPair(first, 0, second, 6) );
    required_struct.addPair( makeConnectPair(first, 1, second, 5) );
    required_struct.addPair( makeConnectPair(first, 2, second, 4) );
    required_struct.addPair( makeConnectPair(first, 3, second, 3) );
    required_struct.addPair( makeConnectPair(first, 4, second, 2) );
    required_struct.addPair( makeConnectPair(first, 5, second, 1) );
    required_struct.addPair( makeConnectPair(first, 6, second, 0) );

    BOOST_CHECK(result_struct == required_struct);
    BOOST_CHECK_EQUAL(p.getSecStructEnergy(),17);
}

BOOST_AUTO_TEST_CASE( FoldedPairLoopsTest ) {
    const int MAX_FOLDINGS = 1000;
    EnergyNucleo energy = createDefaultEnergy();

    FoldedPair p( Chain(string("AAAAAACCCCTTTTGGGGAAAAAA")),
                  Chain(string("TTTTTTTTTTTT")),
                  energy,
                  MAX_FOLDINGS );

    BOOST_REQUIRE( p.getStructures().size() > 0 );

    //  cout << p << endl;
    //  p.printMatrix(cout,3);
    //  p.printStructures(cout,3);

    BOOST_CHECK( p.getStructures().size() == 1 );
    const SecStruct& result_struct = *(p.getStructures().begin() );

    const Chain& first = p.getFirstChain();
    const Chain& second = p.getSecondChain();

    SecStruct required_struct;
    required_struct.addPair( makeConnectPair(first, 0, second,11) );
    required_struct.addPair( makeConnectPair(first, 1, second,10) );
    required_struct.addPair( makeConnectPair(first, 2, second, 9) );
    required_struct.addPair( makeConnectPair(first, 3, second, 8) );
    required_struct.addPair( makeConnectPair(first, 4, second, 7) );
    required_struct.addPair( makeConnectPair(first, 5, second, 6) );

    required_struct.addPair( makeConnectPair(first, 6, first, 17) );
    required_struct.addPair( makeConnectPair(first, 7, first, 16) );
    required_struct.addPair( makeConnectPair(first, 8, first, 15) );
    required_struct.addPair( makeConnectPair(first, 9, first, 14) );

    required_struct.addPair( makeConnectPair(first,18, second, 5) );
    required_struct.addPair( makeConnectPair(first,19, second, 4) );
    required_struct.addPair( makeConnectPair(first,20, second, 3) );
    required_struct.addPair( makeConnectPair(first,21, second, 2) );
    required_struct.addPair( makeConnectPair(first,22, second, 1) );
    required_struct.addPair( makeConnectPair(first,23, second, 0) );


    BOOST_CHECK(result_struct == required_struct);
    BOOST_CHECK_EQUAL(p.getSecStructEnergy(), 36);

}

BOOST_AUTO_TEST_CASE( CodonTest ) {
    //tworzenie - bledy
    BOOST_CHECK_NO_THROW( Codon("AcG") );
    BOOST_CHECK_THROW( Codon("AT"), CodonStringTooShortException );
    BOOST_CHECK_THROW( Codon("AAZ"), NucleotideBadCharException );

    //tworzenie poprawne
    Codon cod1("AAA");
    Codon cod2;

    Nucleotide a(ADENINE);
    Nucleotide t(THYMINE);
    Nucleotide g(GUANINE);
    Nucleotide c(CYTOSINE);

    Codon cod3(a,a,a);

    //operator==
    BOOST_CHECK( cod1 == cod3 );
    BOOST_CHECK( cod1 != cod2 );

    //dostep do skladowych
    BOOST_CHECK( cod1.getFirst() == a );
    BOOST_CHECK( cod1.getSecond() == a );
    BOOST_CHECK( cod1.getThird() == a );

    //operator<
    BOOST_CHECK( ! (cod1 < cod3) );
    BOOST_CHECK( ! (cod3 < cod1) );

    ostringstream s;
    s << cod1;
    BOOST_CHECK_EQUAL( s.str(), string("(A,A,A)") );
}

BOOST_AUTO_TEST_CASE( CodonAminoTableTest ) {
    CodonAminoTable& table = CodonAminoTable::getInstance();

    BOOST_CHECK_EQUAL( table.getAmino(Codon("AAA")),LYSINE );
    BOOST_CHECK_EQUAL( table.getAmino(Codon("AAC")),ASPARAGINE );
    BOOST_CHECK_EQUAL( table.getAmino(Codon("AAT")),ASPARAGINE );
    BOOST_CHECK_EQUAL( table.getAmino(Codon("AAG")),LYSINE );

    set<Codon> set1 = table.getCodons( PHENYLALANINE );
    set<Codon> set1_ref; set1_ref.insert(Codon("TTT")); set1_ref.insert(Codon("TTC"));
    BOOST_CHECK_EQUAL_COLLECTIONS( set1.begin(), set1.end(), set1_ref.begin(), set1_ref.end() );

    set<Codon> set2 = table.getCodons( LEUCINE );
    BOOST_CHECK_EQUAL( set2.size(), 6UL );
    set<Codon> set2_ref = table.getCodons(Codon("TTA"));
    BOOST_CHECK_EQUAL_COLLECTIONS(set2.begin(),set2.end(), set2_ref.begin(), set2_ref.end() );

    set<Codon> set3 = table.getCodons( METHIONINE );
    set<Codon> set3_ref; set3_ref.insert(Codon("ATG"));
    BOOST_CHECK_EQUAL_COLLECTIONS( set3.begin(), set3.end(), set3_ref.begin(), set3_ref.end() );
}

BOOST_AUTO_TEST_SUITE_END()
