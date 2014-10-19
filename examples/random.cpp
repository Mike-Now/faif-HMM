#include <iostream>
#include <faif/utils/Random.hpp>

using namespace faif;

int main() {

	RandomDouble r1; //initializes the random generator, creates the object
	std::cout <<r1() << std::endl; //uniform distribution in range <0,1)

	RandomInt r2(1,6); //the uniform generator from min to max, {1,2,3,4,5,6}
	std::cout << r2() << std::endl;
	return 0;
}
