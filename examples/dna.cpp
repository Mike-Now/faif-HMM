#include <iostream>
#include <string>
#include <set>

#include <faif/dna/Chain.h>
#include <faif/dna/EnergyNucleo.h>
#include <faif/dna/FoldedChain.h>
#include <faif/dna/FoldedPair.h>
#include <faif/dna/Codon.h>
#include <faif/dna/CodonAminoTable.h>

using namespace faif::dna;

int main() {
	Chain x("AAAAAAGGGTTTTTT"); //dna chain
	EnergyNucleo energy = createDefaultEnergy(); //the energy matrix
	// the default values are 2 for A-T, 3 for G-C
	std::cout << (energy.getEnergy( Nucleotide(ADENINE), Nucleotide(THYMINE) ) == 2) << std::endl;

	FoldedChain folded( x, energy );

	std::cout << "maximum energy=" << folded.getSecStructEnergy() << std::endl; //the maximum energy
	SecStructures result = folded.getStructures(); //the structures
	if(!result.empty())
		std::cout << *(result.begin()) << std::endl;

	FoldedPair p( Chain(std::string("AAAAAGGG")), Chain(std::string("GGGTTTTT")), energy, 10 ); //maximum 10 second structures
	if(p.getStructures().size() > 0) {
		const SecStruct& result_struct = *(p.getStructures().begin() ); //get the first structure
		std::cout << result_struct << std::endl;
	}

	Codon cod1("AAA");
	CodonAminoTable& table = CodonAminoTable::getInstance();
	std::cout << "Amino for codon " << cod1 << " is " << table.getAmino( cod1 )  << std::endl; //read the amino for codon
	std::cout << "The no of codons for amino PHENYLANE is " << table.getCodons( PHENYLALANINE ).size() << std::endl; //read the codons for amino
	return 0;
}
