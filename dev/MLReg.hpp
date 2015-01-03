#ifndef FAIF_MLReg_HPP
#define FAIF_MLReg_HPP

#include <string>
#include <boost/unordered_map.hpp>
#include <boost/functional/factory.hpp>
#include <boost/functional/value_factory.hpp>
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
                    };
                    class GISTraining : public MLRegTraining{
                        public:
                            void train(const ExamplesTrain& e);
                    };
                private:
                    class TrainingFactory{
                        private:
                            static TrainingFactory instance;
                            boost::unordered_map<std::string,boost::value_factory<MLRegTraining>> trainings;
                        public:
                            static TrainingFactory& getInstance(){
                                return instance;
                            }
                            boost::value_factory<MLRegTraining> getTraining(std::string trainingId);
                            template<class T>
                                void registerTraining(std::string trainingId);
                        private:
                            TrainingFactory();
                            ~TrainingFactory();
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
            MLReg<Val>::MLReg(const Domains& attr_domains, const AttrDomain& category_domain,std::string algorithmId)
            : Classifier<Val>(attr_domains, category_domain)
            {
                this->reset(algorithmId);
            }

        /** clear the learned parameters */
        template<typename Val>
            void MLReg<Val>::reset() {
                this->reset(currentTrainingId);
            };

        template<typename Val>
            void MLReg<Val>::reset(std::string algorithmId){
                boost::value_factory<MLRegTraining> factory;
                factory = TrainingFactory::getInstance().getFactory(algorithmId);
                trainingImpl.reset(factory());
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
            boost::value_factory<typename MLReg<Val>::MLRegTraining>
            MLReg<Val>::TrainingFactory::
            getTraining(std::string trainingId){

                /* typedef boost::function<MLRegTraining*()> TrainingFactory; */
                /* static std::map<std::string,TrainingFactory> Factories; */
            }

        template<typename Val>
            template<class T>
            void MLReg<Val>::TrainingFactory::
            registerTraining(std::string trainingId)
            {
                const std::unique_ptr<boost::value_factory<T>> factory;

                this->trainings.insert(factory);
            }
    }
}

#endif
