#ifndef FAIF_MLReg_HPP
#define FAIF_MLReg_HPP

#include <string>
#include <map>
#include <memory>
#include <boost/functional/factory.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/static_assert.hpp>
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

                    typedef std::unique_ptr< MLRegTraining> MLRegTrainingPtr;
                    typedef boost::function< MLRegTrainingPtr ()> AbstractFactory;
                private:
                    class TrainingFactory;
                    std::unique_ptr<MLRegTraining>trainingImpl;
                    std::string currentTrainingId;
                public:
                    MLReg();
                    MLReg(const Domains& attr_domains, const AttrDomain& category_domains,std::string algorithmId );
                    virtual ~MLReg() { }

                    virtual void reset();
                    virtual void reset(std::string algorithmId);

                    std::string getTrainingName(){return currentTrainingId;}

                    static void registerTraining(std::string algName,TrainingFactory factory);

                    AttrIdd getCategory(const ExampleTest& example) const;

                    Beliefs getCategories(const ExampleTest& example) const{Beliefs b; return b;}

                    /** \brief train classifier */
                    virtual void train(const ExamplesTrain& e) {
                        trainingImpl->train(e);
                    }

                    /** the ostream method */
                    virtual void write(std::ostream& os) const;

                    class MLRegTraining {
                        public:
                            virtual void train(const ExamplesTrain& e)=0;
                            virtual ~MLRegTraining(){};
                    };
                    class GISTraining : public MLRegTraining{
                        public:
                            void train(const ExamplesTrain& e){}
                            ~GISTraining(){}
                    };
                private:
                    class TrainingFactory{
                        private:
                            typedef typename MLReg<Val>::AbstractFactory AbstractFactory;
                            std::map<std::string,AbstractFactory> trainings;
                        public:
                            TrainingFactory();
                            static TrainingFactory& getInstance(){

                                static TrainingFactory instance;
                                return instance;
                            }
                            AbstractFactory getFactory(std::string trainingId);
                            template<class T>
                                void registerTraining(std::string trainingId);
                        private:
                            ~TrainingFactory(){}
                            TrainingFactory(const TrainingFactory&t);
                            TrainingFactory& operator=(const TrainingFactory&);
                    };
            }; //class MLReg

        //////////////////////////////////////////////////////////////////////////////////////////////////
        // class MLReg implementation
        //////////////////////////////////////////////////////////////////////////////////////////////////
        template<typename Val>
            MLReg<Val>::MLReg() : Classifier<Val>()
        {
        }

        template<typename Val>
            MLReg<Val>::MLReg(const Domains& attr_domains, const AttrDomain& category_domain,std::string trainingId)
            : Classifier<Val>(attr_domains, category_domain)
            {
                currentTrainingId = trainingId;
                this->reset(trainingId);
            }

        /** clear the learned parameters */
        template<typename Val>
            void MLReg<Val>::reset() {
                this->reset(currentTrainingId);
            };

        template<typename Val>
            void MLReg<Val>::reset(std::string treningId){
                AbstractFactory factory;
                factory = TrainingFactory::getInstance().getFactory(treningId);
                trainingImpl=factory();
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

        //////////////////////////////////////////////////////////////////////////////////////////////////
        // class TrainingFactory implementation
        //////////////////////////////////////////////////////////////////////////////////////////////////
        template<typename Val>
            MLReg<Val>::TrainingFactory::TrainingFactory(){
                registerTraining<MLReg<Val>::GISTraining>("GIS");
            }
        template<typename Val>
            typename MLReg<Val>::AbstractFactory
            MLReg<Val>::TrainingFactory::
            getFactory(std::string trainingId){
                typename std::map<std::string, AbstractFactory>::iterator it;
                it = trainings.find(trainingId);
                if(it != trainings.end())
                {
                    return it->second;
                }
                else{
                    std::cout<<"IS NULL PANIC";
                    return NULL; //TODO generalized exception
                }
            }

        template<typename Val>
            template<class T>
            void MLReg<Val>::TrainingFactory::
            registerTraining(std::string trainingId)
            {
                BOOST_STATIC_ASSERT_MSG(
                        (std::is_base_of<MLRegTraining, T>::value),
                        "A registered training must be a descendant of MLRegTraining."
                        );
                MLReg<Val>::AbstractFactory factory = boost::factory<std::unique_ptr<T>>();
                this->trainings.insert(std::make_pair(trainingId,factory));

            }
    }
}

#endif
