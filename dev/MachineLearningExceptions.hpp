#ifndef FAIF_ML_EXCEPTIONS
#define FAIF_ML_EXCEPTIONS

#include "../ExceptionsFaif.hpp"

namespace faif {
    namespace ml{

        class MissingTrainingException : public FaifException{
            std::string trainingId;
            public:
            MissingTrainingException(std::string trainingId){
                this->trainingId=trainingId;
            }
            virtual ~MissingTrainingException() throw() {}
            virtual const char *what() const throw(){return "MissingTrainingException";}
            virtual std::ostream& print(std::ostream &os) const throw(){
                os << "The training class: "<<trainingId<<" was not found."<<std::endl;
                return os;
            }
        };
    }
}
#endif
