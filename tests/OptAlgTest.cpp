/** the test of optimization algorithms (e.g. evolutionary algorithm) */

#if defined(_MSC_VER) && (_MSC_VER >= 1400)
//msvc9.0 warning when while(1)
#pragma warning(disable:4127)
//msvc9.0 warning boost::concept_check EqualityComparable
#pragma warning(disable:4510)
//msvc9.0 generuje smieci dla boost/random
#pragma warning(disable:4512)
#pragma warning(disable:4610)
#endif

#include <iostream>
#include <algorithm>
#include <iterator>
#include <numeric>

#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/array.hpp>

#include "../src/search/EvolutionaryAlgorithm.hpp"
#include "../src/search/VectorIndividual.hpp"
#include "../src/search/ExpectationMaximization.hpp"
#include "../src/search/HillClimbing.hpp"
#include "../src/utils/Random.hpp"


using namespace std;
using namespace faif;
using namespace faif::search;
using namespace boost;
using boost::unit_test::test_suite;

BOOST_AUTO_TEST_SUITE( Faif_optimization_algorithms_test )

namespace {

    /** \brief class to test evolutionary algorithm to minimize square(x) */
    class EASpace  : public EvolutionaryAlgorithmSpace<double> {
    public:
        //custom mutation
        static Individual& mutation(Individual& ind) {
            RandomDouble r;
            ind = 4.0*(r() - 0.5)*ind;
            return ind;
        }

		static Individual& crossover(Individual& ind, Population& pop) {
			RandomInt r(0, static_cast<int>(pop.size()));
			int idx = r();
			if( idx >= static_cast<int>(pop.size()) )
				idx = 0;
			ind = (ind + pop[idx]) / 2.0;
			return ind;
		}
        //the fitness function
        static double fitness(const Individual& ind) { return -ind*ind; }
    };

} //namespace


/** easy test - find the maximum of -(x*x) function */
BOOST_AUTO_TEST_CASE( EAMaximumTest1 ) {
    boost::array<double,4> D  = { { 1.0, -2.0, -4.0, 8.0 } };
	EASpace::Population init_population( D.begin(), D.end() );
    EvolutionaryAlgorithm<EASpace, MutationCustom, CrossoverNone, SelectionRanking, StopAfterNSteps<30> > ea;
    BOOST_CHECK_SMALL( ea.solve(init_population), 0.1 );
}

/** easy test - find the maximum of -(x*x) function */
BOOST_AUTO_TEST_CASE( EAMaximumTest2 ) {
    RandomDouble gen;
    vector<double> random_init(3);
    generate( random_init.begin(), random_init.end(), gen );
    EvolutionaryAlgorithm<EASpace, MutationCustom, CrossoverNone, SelectionRanking, StopAfterNSteps<30> > ea;
    BOOST_CHECK_SMALL( ea.solve(random_init), 0.1 );
}

/** easy test - find the maximum of -(x*x) function */
BOOST_AUTO_TEST_CASE( EAMaximumTest3 ) {
    boost::array<double,4> D  = { { 1.0, -2.0, -4.0, 8.0 } };
	EASpace::Population init_population( D.begin(), D.end() );
    EvolutionaryAlgorithm<EASpace, MutationCustom, CrossoverNone, SelectionRoulette, StopAfterNSteps<3000> > ea;
    BOOST_CHECK_SMALL( ea.solve(init_population), 0.1 );
}

/** easy test - find the maximum of -(x*x) function */
BOOST_AUTO_TEST_CASE( EAMaximumTest4 ) {
    boost::array<double,4> D  = { { 1.0, -2.0, -4.0, 8.0 } };
	EASpace::Population init_population( D.begin(), D.end() );
    EvolutionaryAlgorithm<EASpace, MutationNone, CrossoverCustom, SelectionRanking, StopAfterNSteps<3000> > ea;
    BOOST_CHECK_SMALL( ea.solve(init_population), 0.1 );
}

namespace {
    /** \brief class to test the backpack algorithm */

    class Backpack {
    public:
        /** \brief creates the empty backpack */
        Backpack(const vector<int>& all, int max_backpack)
            : left_(all), max_(max_backpack)
        { }

        ~Backpack() {}

        /** \brief remove one or add one element to backpack */
        void mutation() {

            RandomInt r(0,1);

            if( r() == 0 ) {

                if( left_.empty() )
                    return;
                //adds a random element
                RandomInt ri(0, left_.size() - 1 );

                vector<int>::iterator rand_it = left_.begin();
                int r = ri();
                advance( rand_it, r );
                if(fitness() + *rand_it <= max_) {
                    backpack_.push_back( *rand_it );
                    left_.erase( rand_it );
                }
            }
            else {
                if( backpack_.empty() )
                    return;

                //removes a random  element
                RandomInt ri(0, backpack_.size() - 1 );
                vector<int>::iterator rand_it = backpack_.begin();
                advance(rand_it, ri() );
                backpack_.erase(rand_it);
            }
        }
        int fitness() const { return accumulate(backpack_.begin(), backpack_.end(), 0); }

        friend ostream& operator<<(ostream& os, const Backpack& back);

    private:
        vector<int> backpack_;
        vector<int> left_;
        int max_;
    };

    ostream& operator<<(ostream& os, const Backpack& back) {
        os << "backpack: ";
        copy(back.backpack_.begin(), back.backpack_.end(), ostream_iterator<int>(os,",") );
        os << "  left: ";
        copy(back.left_.begin(), back.left_.end(), ostream_iterator<int>(os,",") );
        os << "  max: " << back.max_ << endl;
        return os;
    }

    struct BackpackSpace : public EvolutionaryAlgorithmSpace<Backpack> {

        static Individual& mutation(Individual& ind) {
            ind.mutation();
            return ind;
        }

        static int fitness(const Individual& ind) { return ind.fitness(); }
    };

}

/** test - solve the backpack problem */
void EABackpackTest(vector<int> backpack, int max_backpack) {

    const int POPULATION_SIZE = 10;
    EvolutionaryAlgorithm<BackpackSpace, MutationCustom, CrossoverNone, SelectionRanking, StopAfterNSteps<50> > ea;

    while(1) {
        int i = 0;
        BackpackSpace::Population pop(POPULATION_SIZE, Backpack(backpack, max_backpack) );
        Backpack& best = ea.solve(pop);
        if(best.fitness() == max_backpack)
            break;
        ++i;
        if(i >= 10) {
            BOOST_ERROR("not found the solution in backpack, for backpack size= " << max_backpack << "(10 times try)");
            break;
        }
    }
}


BOOST_AUTO_TEST_CASE( BackpackEATest13 ) {
    boost::array<int,6> TAB  = { { 2, 3, 4, 5, 9, 11 } };
    vector<int> backpack(TAB.begin(), TAB.end() );
    EABackpackTest(backpack, 13);
}

BOOST_AUTO_TEST_CASE( BackpackEATest14 ) {
    boost::array<int,6> TAB  = { { 2, 3, 4, 5, 9, 11 } };
    vector<int> backpack(TAB.begin(), TAB.end() );
    EABackpackTest(backpack, 14);
}

BOOST_AUTO_TEST_CASE( BackpackEATest15 ) {
    boost::array<int,6> TAB  = { { 2, 3, 4, 5, 9, 11 } };
    vector<int> backpack(TAB.begin(), TAB.end() );
    EABackpackTest(backpack, 15);
}

BOOST_AUTO_TEST_CASE( BackpackEATest16 ) {
    boost::array<int,6> TAB  = { { 2, 3, 4, 5, 9, 11 } };
    vector<int> backpack(TAB.begin(), TAB.end() );
    EABackpackTest(backpack, 16);
}

BOOST_AUTO_TEST_CASE( BackpackEATest17 ) {
    boost::array<int,6> TAB  = { { 2, 3, 4, 5, 9, 11 } };
    vector<int> backpack(TAB.begin(), TAB.end() );
    EABackpackTest(backpack, 17);
}

BOOST_AUTO_TEST_CASE( BackpackEATest18 ) {
    boost::array<int,6> TAB  = { { 2, 3, 4, 5, 9, 11 } };
    vector<int> backpack(TAB.begin(), TAB.end() );
    EABackpackTest(backpack, 18);
}

BOOST_AUTO_TEST_CASE( BackpackEATest19 ) {
    boost::array<int,6> TAB  = { { 2, 3, 4, 5, 9, 11 } };
    vector<int> backpack(TAB.begin(), TAB.end() );
    EABackpackTest(backpack, 19);
}

BOOST_AUTO_TEST_CASE( BackpackEATest20 ) {
    boost::array<int,6> TAB  = { { 2, 3, 4, 5, 9, 11 } };
    vector<int> backpack(TAB.begin(), TAB.end() );
    EABackpackTest(backpack, 20);
}

BOOST_AUTO_TEST_CASE( BackpackEATest21 ) {
    boost::array<int,6> TAB  = { { 2, 3, 4, 5, 9, 11 } };
    vector<int> backpack(TAB.begin(), TAB.end() );
    EABackpackTest(backpack, 21);
}

BOOST_AUTO_TEST_CASE( BackpackEATest22 ) {
    boost::array<int,6> TAB  = { { 2, 3, 4, 5, 9, 11 } };
    vector<int> backpack(TAB.begin(), TAB.end() );
    EABackpackTest(backpack, 22);
}

BOOST_AUTO_TEST_CASE( BackpackEATest23 ) {
    boost::array<int,6> TAB  = { { 2, 3, 4, 5, 9, 11 } };
    vector<int> backpack(TAB.begin(), TAB.end() );
    EABackpackTest(backpack, 23);
}

BOOST_AUTO_TEST_CASE( BackpackEATest24 ) {
    boost::array<int,6> TAB  = { { 2, 3, 4, 5, 9, 11 } };
    vector<int> backpack(TAB.begin(), TAB.end() );
    EABackpackTest(backpack, 24);
}

BOOST_AUTO_TEST_CASE( BackpackEATest25 ) {
    boost::array<int,6> TAB  = { { 2, 3, 4, 5, 9, 11 } };
    vector<int> backpack(TAB.begin(), TAB.end() );
    EABackpackTest(backpack, 25);
}

BOOST_AUTO_TEST_CASE( BackpackEATest26 ) {
    boost::array<int,6> TAB  = { { 2, 3, 4, 5, 9, 11 } };
    vector<int> backpack(TAB.begin(), TAB.end() );
    EABackpackTest(backpack, 26);
}

BOOST_AUTO_TEST_CASE( BackpackEATest27 ) {
    boost::array<int,6> TAB  = { { 2, 3, 4, 5, 9, 11 } };
    vector<int> backpack(TAB.begin(), TAB.end() );
    EABackpackTest(backpack, 27);
}

BOOST_AUTO_TEST_CASE( BackpackEATest28 ) {
    boost::array<int,6> TAB  = { { 2, 3, 4, 5, 9, 11 } };
    vector<int> backpack(TAB.begin(), TAB.end() );
    EABackpackTest(backpack, 28);
}

BOOST_AUTO_TEST_CASE( BackpackEATest29 ) {
     boost::array<int,6> TAB  = { { 2, 3, 4, 5, 9, 11 } };
    vector<int> backpack(TAB.begin(), TAB.end() );
    EABackpackTest(backpack, 29);
}

BOOST_AUTO_TEST_CASE( BackpackEATest30 ) {
    boost::array<int,6> TAB  = { { 2, 3, 4, 5, 9, 11 } };
    vector<int> backpack(TAB.begin(), TAB.end() );
    EABackpackTest(backpack, 30);
}


BOOST_AUTO_TEST_CASE( vector_individual_test ) {
    VectorIndividual<BooleanGene> ind(10);
    VectorIndividual<BooleanGene> ind2(ind);
    BOOST_CHECK( ind == ind2 );
    ind.mutate(0.0);
    BOOST_CHECK( ind == ind2 );
    ind.mutate(1.0);
    BOOST_CHECK( ind != ind2 );
    ind2 = ind;
    ind.mutate(0.5);
    BOOST_CHECK( ind != ind2 );
}

BOOST_AUTO_TEST_CASE( vector_individual_custom_test ) {
	VectorIndividual<BooleanGene>::Container cont(10);
	fill( cont.begin(), cont.end(), false );
	VectorIndividual<BooleanGene> ind( cont );
	const VectorIndividual<BooleanGene>::Container& ind_cont = ind.getChromosome();
    BOOST_CHECK_EQUAL_COLLECTIONS( cont.begin(), cont.end(), ind_cont.begin(), ind_cont.end() );
}

namespace {

    /** \brief class to test evolutionary algorithm to minimize square(x) */
    typedef VectorIndividual<BooleanGene> MaxBinIndividual;
    struct MaxBinSpace  : public EvolutionaryAlgorithmSpace<MaxBinIndividual> {
    public:
        static Individual& mutation(Individual& ind) {         //custom mutation
            ind.mutate(0.1);
            return ind;
        }
        static double fitness(const Individual& ind) {         //the fitness function
            int count = 0;
            for( Individual::Container::const_iterator it = ind.getChromosome().begin();
                 it != ind.getChromosome().end(); ++it )
                if( *it ) ++count;
            return static_cast<double>(count);
        }
    };
}

BOOST_AUTO_TEST_CASE( max_bin_space ) {
    MaxBinSpace::Population population;
    population.push_back( MaxBinSpace::Individual(10) );
    population.push_back( MaxBinSpace::Individual(10) );

    EvolutionaryAlgorithm<MaxBinSpace, MutationCustom, CrossoverNone, SelectionRanking, StopAfterNSteps<500> > ea;
    MaxBinSpace::Individual ind = ea.solve( population );
    BOOST_CHECK( MaxBinSpace::fitness(ind) == 10 );
}

namespace {
    /** \brief the class with user stop condition for evolutinary algorithm */
    struct MyOwnStopCondition {
        MyOwnStopCondition(int num) : steps_(0), num_(num) { }
        template<typename Population> void update(const Population& ) { ++steps_; }
        bool isFinished() const { return steps_ >= num_; }
    private:
        unsigned int steps_;
        unsigned int num_;
    };
}

BOOST_AUTO_TEST_CASE( test_ea_own_stop ) {

    MaxBinSpace::Population population;
    population.push_back( MaxBinSpace::Individual(10) );
    population.push_back( MaxBinSpace::Individual(10) );

    EvolutionaryAlgorithm<MaxBinSpace, MutationCustom, CrossoverNone, SelectionRanking, MyOwnStopCondition > ea;
    MyOwnStopCondition stop(100);
    MaxBinSpace::Individual ind = ea.solve( population, stop );
    BOOST_CHECK( MaxBinSpace::fitness(ind) == 10 );

}

namespace {

	typedef double EMParam;

    /** very simple space, calculates x := (x*x + 1 )/2, it to 1.0  */
	struct EMSpace : public Space<EMParam> {

        static Individual& expectation(Individual& x) {
            x = x*x + 1.0;
            return x;
        }
        static Individual& maximization(Individual& x) {
			x /= 2.0;
            return x;
        }
		static Fitness fitness(Individual&) {
			return 0.0;
		}
	};

} //namespace

BOOST_AUTO_TEST_CASE( EMSimpleTest ) {

    ExpectationMaximization<EMSpace, ExpectationCustom, MaximizationCustom, StopAfterNSteps<1000> > em;
	EMSpace::Individual init(0);
	EMSpace::Individual result = em.solve(init);
    BOOST_CHECK_CLOSE( 1.0, result, 1.0 );
}

namespace {

    struct HillClimbInd : public Node<HillClimbInd> {

        HillClimbInd(double v) : val_(v), step_(v) { /* cout << "ctor " << val_ << endl; */ }
        HillClimbInd(double v, double s) : val_(v), step_(s) { /* cout << "ctor " << val_ << endl; */ }
        ~HillClimbInd() { /* cout << "dtor " << val_ << endl; */ }

        bool operator==(const HillClimbInd& i) const { return val_ == i.val_; }
        bool operator!=(const HillClimbInd& i) const { return val_ != i.val_; }

        Children getChildren() const {
            Children children;
            children.push_back( PNode(new HillClimbInd(val_ + step_, step_ / 2.0) ) );
            children.push_back( PNode(new HillClimbInd(val_ - step_, step_ / 2.0) ) );
            return children;
        }
        double val_;
        double step_;
    };


    /** \brief class to test hill climbing to maximize function f(x) = -1.0 * square(x) */
    struct HillClimbSpace : public Space<HillClimbInd> {
        static Fitness fitness(const Individual& ind) {
            /* cout << "fitness for " << ind.val_ << endl; */
            return -ind.val_*ind.val_;
        }
    };

}

BOOST_AUTO_TEST_CASE( HillClimbTest ) {
    HillClimbing<HillClimbSpace> alg;
    HillClimbSpace::Individual::PNode ind(new HillClimbSpace::Individual(100) );
    HillClimbSpace::Individual::PNode result =  alg.solve(ind);
    BOOST_CHECK_CLOSE( result->val_, 0.0, 0.1 );
}

BOOST_AUTO_TEST_SUITE_END()
