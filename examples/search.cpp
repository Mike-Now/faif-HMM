#include <iostream>
#include <algorithm>
#include <iterator>
#include <boost/bind.hpp>
#include <faif/search/DepthFirst.h>
#include <faif/search/BreadthFirst.h>
#include <faif/search/UnifiedCost.h>
#include <faif/search/AStar.h>

using namespace faif;
using namespace faif::search;


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

	friend std::ostream& operator<<(std::ostream& os, const SimpleTree& s);
private:
	int n_;
};

std::ostream& operator<<(std::ostream& os, const SimpleTree& s) {
	os << s.n_;
	return os;
}

int main() {

    SimpleTree::PNode root(new SimpleTree() );
    {
        SimpleTree::Path p = searchDepthFirst<SimpleTree>( root );
		std::cout << p << std::endl; //p is the path of nodes: 10-5-2-1
    }
    {
        SimpleTree::Path p = searchBreadthFirst<SimpleTree>( root );
        std::cout << p << std::endl; //p is the path of nodes: 10-5-2-1
    }
    {
        SimpleTree::Path p = searchUnifiedCost<SimpleTree>( root );
        std::cout << p << std::endl; //p is the path of nodes: 10-5-2-1
    }
    {
        SimpleTree::Path p = searchAStar<SimpleTree>( root );
        std::cout << p << std::endl; //p is the path of nodes: 10-5-2-1
    }
    return 0;
}
