/** the test of searching methods */

#if defined(_MSC_VER) && (_MSC_VER >= 1400)
//msvc9.0 warning boost::concept_check EqualityComparable
#pragma warning(disable:4510)
#pragma warning(disable:4610)
#endif

#include <iostream>
#include <algorithm>

#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "../src/search/DepthFirst.h"
#include "../src/search/BreadthFirst.h"
#include "../src/search/UnifiedCost.h"
#include "../src/search/AStar.h"

using namespace std;
using namespace faif;
using namespace faif::search;
using namespace boost;
using boost::unit_test::test_suite;

BOOST_AUTO_TEST_SUITE( Faif_search_test )


namespace {

    /** class to test searching method

        1---------------------
        |                     \
        2                      3
        | \                    |\
        4  5                   6 7
        |\  \
        8 9  10 (FINAL)

    */
    class SimpleTree : public Node<SimpleTree> {
    public:
        static const int MAX_N = 10;

        SimpleTree(int n = 1) : n_(n) {}
        ~SimpleTree() {}

        bool operator==(const SimpleTree& t) const { return n_ == t.n_; }
        bool operator!=(const SimpleTree& t) const { return n_ != t.n_; }

        bool isFinal() const { return n_ == MAX_N; }

        int getN() const { return n_; }

        vector<boost::shared_ptr<SimpleTree> > getChildren() {
            Children children;
            if(2*n_ <= MAX_N)
                children.push_back( PNode(new SimpleTree(2*n_) ) );
            if(2*n_ + 1 <= MAX_N)
                children.push_back( PNode(new SimpleTree(2*n_ + 1) ) );
            return children;
        }

        /** method required by informed search algorithms (AStar, UnifiedCost) */
        double getWeight() const { return 1.0; }
        /** method required by heuristic search algorithms (AStar) */
        double getHeuristic() const { return 1.0; }
    private:
        int n_;
    };

    ostream& operator<<(ostream& os, const SimpleTree& s) {
        os << s.getN();
        return os;
    }

    /** check if path is proper */
    void check_proper(const Node<SimpleTree>::Path& p) {
        BOOST_REQUIRE_EQUAL( p.size(), 4U );
        BOOST_CHECK_EQUAL( p[0]->getN(), 10 );
        BOOST_CHECK_EQUAL( p[1]->getN(), 5 );
        BOOST_CHECK_EQUAL( p[2]->getN(), 2 );
        BOOST_CHECK_EQUAL( p[3]->getN(), 1 );
    }

    /** \brief class to test searching method with recursive paths

        1---------------------
        |                     \
        2                      3
        | \                    |\
        4  5                   6 7
        |\  \                  | |
        8 9  10 (FINAL)        1 1
        | |  |
        1 1  1
    */
    struct SimpleRecTree : public Node<SimpleRecTree> {
    public:
        static const int MAX_N = 10;

        SimpleRecTree(int n , PNode r ) : n_(n), root_(r) {}
        ~SimpleRecTree() {}

        bool operator==(const SimpleRecTree& t) const { return n_ == t.n_; }
        bool operator!=(const SimpleRecTree& t) const { return n_ != t.n_; }

        bool isFinal() const { return n_ == MAX_N; }

        /** method required by informed search algorithms (AStar, UnifiedCost) */
        double getWeight() const { return 1.0; }
        /** method required by heuristic search algorithms (AStar) */
        double getHeuristic() const { return 1.0; }

        Children getChildren() {
            Children children;
            if(2*n_ <= MAX_N)
                children.push_back( PNode(new SimpleRecTree(2*n_, root_ ) ) );
            if(2*n_ + 1 <= MAX_N)
                children.push_back( PNode(new SimpleRecTree(2*n_ + 1, root_ ) ) );
            if( children.empty() ) {
                children.push_back( root_ );
            }
            return children;
        }

        int n_;
        PNode root_;

    };

    ostream& operator<<(ostream& os, const SimpleRecTree& s) {
        os << s.n_;
        return os;
    }

    /** check if path is proper */
    void check_proper(const Node<SimpleRecTree>::Path& p) {
        BOOST_REQUIRE_EQUAL( p.size(), 4U );
        BOOST_CHECK_EQUAL( p[0]->n_, 10 );
        BOOST_CHECK_EQUAL( p[1]->n_, 5 );
        BOOST_CHECK_EQUAL( p[2]->n_, 2 );
        BOOST_CHECK_EQUAL( p[3]->n_, 1 );
    }

}



BOOST_AUTO_TEST_CASE( DepthFirstTest ) {

    SimpleTree::PNode root(new SimpleTree() );
    SimpleTree::Path p = searchDepthFirst<SimpleTree>( root );

    check_proper(p);
}

BOOST_AUTO_TEST_CASE( BreadthFirstTest ) {

    SimpleTree::PNode root(new SimpleTree() );
    SimpleTree::Path p = searchBreadthFirst<SimpleTree>( root );

    check_proper(p);
}

BOOST_AUTO_TEST_CASE( UnifiedCostTest ) {

    SimpleTree::PNode root(new SimpleTree() );
    SimpleTree::Path p = searchUnifiedCost<SimpleTree>( root );

    check_proper(p);
}

BOOST_AUTO_TEST_CASE( AStarTest ) {

    SimpleTree::PNode root(new SimpleTree() );
    SimpleTree::Path p = searchAStar<SimpleTree>( root );

    check_proper(p);
}


BOOST_AUTO_TEST_CASE( DepthFirstRecTest ) {

    SimpleRecTree* raw_root = new SimpleRecTree(1, SimpleRecTree::PNode() );
    SimpleRecTree::PNode root(raw_root);
    raw_root->root_ = root;

    SimpleRecTree::Path p = searchDepthFirst<SimpleRecTree>( root );

    raw_root->root_ = SimpleRecTree::PNode( );

    check_proper(p);
}

BOOST_AUTO_TEST_CASE( BreadthFirstRecTest ) {

    SimpleRecTree* raw_root = new SimpleRecTree(1, SimpleRecTree::PNode() );
    SimpleRecTree::PNode root(raw_root);
    raw_root->root_ = root;

    SimpleRecTree::Path p = searchBreadthFirst<SimpleRecTree>( root );

    raw_root->root_ = SimpleRecTree::PNode( );

    check_proper(p);
}

BOOST_AUTO_TEST_CASE( UnifiedCostRecTest ) {

    SimpleRecTree* raw_root = new SimpleRecTree(1, SimpleRecTree::PNode() );
    SimpleRecTree::PNode root(raw_root);
    raw_root->root_ = root;

    SimpleRecTree::Path p = searchUnifiedCost<SimpleRecTree>( root );

    raw_root->root_ = SimpleRecTree::PNode( );

    check_proper(p);
}


BOOST_AUTO_TEST_CASE( AStarRecTest ) {

    SimpleRecTree* raw_root = new SimpleRecTree(1, SimpleRecTree::PNode() );
    SimpleRecTree::PNode root(raw_root);
    raw_root->root_ = root;

    SimpleRecTree::Path p = searchAStar<SimpleRecTree>( root );

    raw_root->root_ = SimpleRecTree::PNode( );

    check_proper(p);
}

/**
   Klasa testowa do biblioteki searching. Implementuje matematyczn± uk³adankê
   8-puzzle. Do rozwi±zania puzzli testuje siê funkcje searchDepthFirst,
   searchDepthFirstLowMem, searchBreadthFirst i searchBreadthFirstOptimize.
   W wyniku testu otrzymuje siê rozwi±zanie (o ile istnieje) oraz czas
   potrzebny do rozwi±zania (tics/sekundy).
*/


namespace {
    /**
       \brief the 9-puzzle; to test the search algorithms
    */
    class Puzzle: public Node<Puzzle> {
    public:
        static const int SIZE = 9; //the puzzle size (num of puzzle)
        static const int SSIZE = 3; //the puzzle border size (sqrt(SIZE) )

        static bool positionInBorder(int x, int y) {
            return (x >= 0) && (x < SSIZE) && (y >= 0) && (y < SSIZE);
        }

        static int getIdx(int x, int y) {
            return x + SSIZE * y;
        }
    public:
        /** c-tor with starting puzzle */
        Puzzle(char table[SIZE]) {
            copy( table, table + SIZE, state_ );
        }

        Puzzle(const Puzzle& p) {
            copy( p.state_, p.state_ + SIZE, state_ );
        }

        ~Puzzle() { }

        bool operator==(const Puzzle& p) const {
            return equal( state_, state_ + SIZE, p.state_ );
        }

        bool operator!=(const Puzzle& p) const {
            return ! operator==(p);
        }

        /** required by searchUnifiedCost and searchAStar */
        double getWeight() const {
            return 1.0;
        }

		/** count the mismatches */
		double getHeuristic() const {

			double miss = 0.0;

			for(int i=0;i<SIZE;++i)
				if( state_[i] != finish_[i] )
					miss += 1.0;

			return miss;
		}

        /**
           \brief check if final state (compare with the finish_ table)
        */
        bool isFinal() const {
            return equal( state_, state_ + SIZE, finish_ );
        }

        /**
           \brief the next puzzle states
        */
        Children getChildren() {

            Children children;

            //find the index of empty element
            int pos = static_cast<int>( find(state_, state_ + SIZE, ' ') - state_ );

            int x = pos % SSIZE;
            int y = pos / SSIZE;

            if( positionInBorder(x - 1, y) )
                children.push_back( PNode(new Puzzle( createBySwap( getIdx(x - 1, y), pos) ) ) );
            if( positionInBorder(x + 1, y) )
                children.push_back( PNode(new Puzzle( createBySwap( getIdx(x + 1, y), pos) ) ) );
            if( positionInBorder(x, y - 1) )
                children.push_back( PNode(new Puzzle( createBySwap( getIdx(x, y - 1), pos) ) ) );
            if( positionInBorder(x, y + 1) )
                children.push_back( PNode(new Puzzle( createBySwap( getIdx(x, y + 1), pos) ) ) );
            return children;
        }
    private:
        char state_[SIZE];
        static char finish_[SIZE]; //the solved puzzle

        /** new puzzle created by swapping the two cells */
        Puzzle createBySwap(int new_idx, int old_idx) {
            Puzzle n(*this);
            swap( n.state_[new_idx], n.state_[old_idx] );
            return n;
        }


        /** to debug */
        friend ostream& operator<<(ostream& o, const Puzzle& p);
    };

    /** the static member of Puzzle */
    char Puzzle::finish_[Puzzle::SIZE]  = {'1','2','3','4','5','6','7','8',' '};

    ostream& operator<<(ostream& os, const Puzzle& p) {
        for(int i = 0; i < Puzzle::SIZE; ++i) {
            os << p.state_[i];
            if(i % Puzzle::SSIZE == Puzzle::SSIZE - 1)
                os << endl;
        }
        os << endl;
        return os;
    }

	bool compPuzzleSmartPtr(const Puzzle::PNode& a, const Puzzle::PNode& b) {
		return (*a) == (*b);
	}

} //namespace


BOOST_AUTO_TEST_CASE( PuzzleTest ) {

    char start[Puzzle::SIZE] = {'4','5','2',' ','6','1','8','7','3'};
    Puzzle::PNode root(new Puzzle(start) );

    Puzzle::Path p = searchBreadthFirst<Puzzle>( root );
    BOOST_CHECK_EQUAL( p.size(), 20U );

    Puzzle::Path p2 = searchUnifiedCost<Puzzle>( root );
    BOOST_CHECK_EQUAL( p2.size(), 20U );

    Puzzle::Path p3 = searchAStar<Puzzle>( root );
	BOOST_CHECK_EQUAL( p3.size(), 20U );

    //check if the paths are the same
    BOOST_CHECK( equal( p.begin(), p.end(), p2.begin(), compPuzzleSmartPtr ) );
    BOOST_CHECK( equal( p.begin(), p.end(), p3.begin(), compPuzzleSmartPtr ) );
}

/** long test - many backtracks -> so disabled */
// BOOST_AUTO_TEST_CASE( PuzzleDepthTest ) {
//     char start[Puzzle::SIZE] = {'4','5','2',' ','6','1','8','7','3'};
//     Puzzle::PNode root(new Puzzle(start) );
//     Puzzle::Path p = searchDepthFirst<Puzzle>( root );
//     BOOST_CHECK( p.size() > 20U );
// }

BOOST_AUTO_TEST_SUITE_END()
