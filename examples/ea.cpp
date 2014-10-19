#include <faif/search/EvolutionaryAlgorithm.hpp>

using namespace faif;
using namespace faif::search;

//the class represent the problem
class EASpace  : public EvolutionaryAlgorithmSpace<double> {
public:
	//custom mutation
	static Individual& mutation(Individual& ind) {
		RandomDouble r; //returns the double from 0 to 1 with uniform distribution
		ind = 4.0*(r() - 0.5)*ind;
		return ind;
	}
	//the fitness function
	static double fitness(const Individual& ind) { return -ind*ind; }
};

int main() {

	EASpace::Population pop; //initial population
	pop.push_back(1.0);
	pop.push_back(-2.0);
	pop.push_back(-4.0);
	pop.push_back(8.0);

	EvolutionaryAlgorithm< EASpace, MutationCustom, CrossoverNone, SelectionRanking, StopAfterNSteps<30> > ea;
	EASpace::Individual best = ea.solve(pop); //run the ea
	std::cout << best << std::endl;
	return 0;
}
