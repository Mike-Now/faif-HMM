#include <iostream>
#include <faif/search/HillClimbing.hpp>

using namespace faif;
using namespace faif::search;

//the class represent the problem
struct HillClimbInd : public Node<HillClimbInd> {

	HillClimbInd(double v) : val_(v), step_(v) {}
	HillClimbInd(double v, double s) : val_(v), step_(s) {}
	~HillClimbInd() {}

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
		return -ind.val_*ind.val_;
	}
};

int main() {
    HillClimbing<HillClimbSpace> alg;
    HillClimbSpace::Individual::PNode ind(new HillClimbSpace::Individual(100) );
    HillClimbSpace::Individual::PNode result =    alg.solve(ind); //run the algorithm
    std::cout << result->val_ << std::endl;
    return 0;
}
