#ifndef FAIF_MLRegEG_HPP
#define FAIF_MLRegEG_HPP

#include <string>

#include <boost/functional/factory.hpp>
#include <boost/function.hpp>
#include "Classifier.hpp"

namespace faif {
	namespace ml {
/** \brief Multinomial Logistic Regression Classifier
 *
 *
 */
		template<typename Val>
		class MLReg : public Classifier<Val> {
		public:
			typedef typename Classifier<Val>::AttrValue AttrValue;
			typedef typename Classifier<Val>::AttrDomain AttrDomain;
			typedef typename Classifier<Val>::AttrIdd AttrIdd;
			typedef typename Classifier<Val>::AttrIddSerialize AttrIddSerialize;
			typedef typename Classifier<Val>::Domains Domains;
			typedef typename Classifier<Val>::Beliefs Beliefs;
			typedef typename Classifier<Val>::ExampleTest ExampleTest;
			typedef typename Classifier<Val>::ExampleTrain ExampleTrain;
			typedef typename Classifier<Val>::ExamplesTrain ExamplesTrain;
            
            class MLRegTraining;
            class GISTraining; //TMP
        private:
            std::unique_ptr<MLRegTraining>trainingImpl;
            typedef boost::function<MLRegTraining*()> TrainingFactory;
            static std::map<std::string,TrainingFactory> Factories;
            std::string currentAlgorithmId;
        public:
			MLReg();
			MLReg(const Domains& attr_domains, const AttrDomain& category_domains,std::string algorithmId );
			virtual ~MLReg() { }

			virtual void reset();
            
            static void registerAlgorithm(std::string algName,TrainingFactory factory);

            AttrIdd getCategory(const ExampleTest& example) const;

            Beliefs getCategories(const ExampleTest& example) const{Beliefs b; return b;;}

            /** \brief learn classifier (on the collection of training examples) */
            virtual void train(const ExamplesTrain& e) {
                trainingImpl->train(e);
            }

            /** the ostream method */
            virtual void write(std::ostream& os) const;

            ///MLRegTraining///
            //interface
            class MLRegTraining {
            public:
                virtual void train(const ExamplesTrain& e)=0;
            };
            class GISTraining : public MLRegTraining{
            
                void train(const ExamplesTrain& e) {
                    return;
                };
            
            };
        };            
        //class MLReg

        //////////////////////////////////////////////////////////////////////////////////////////////////
        // class MLReg implementation
        //////////////////////////////////////////////////////////////////////////////////////////////////
        template<typename Val>
        MLReg<Val>::MLReg() : Classifier<Val>()
        {
           // impl_.reset( new MLRegTraining(*this) );
           trainingImpl.reset(new GISTraining());
        }

        template<typename Val>
        MLReg<Val>::MLReg(const Domains& attr_domains, const AttrDomain& category_domain,std::string algorithmId)
            : Classifier<Val>(attr_domains, category_domain)
        {
           // impl_.reset( new MLRegTraining(*this) );
           trainingImpl.reset(new GISTraining());
        }

        /** the clear the learned parameters */
        template<typename Val>
        void MLReg<Val>::reset() {
           // impl_.reset( new MLRegTraining(*this) );
           trainingImpl.reset(new GISTraining());
        }; 

        template<typename Val>
        typename MLReg<Val>::AttrIdd
        MLReg<Val>::getCategory(const ExampleTest& example) const {
            return this->getCategoryIdd("good"); //mockup
        };

        /** ostream method */
        template<typename Val>
        void MLReg<Val>::write(std::ostream& os) const {
        }

	}
}

#endif
