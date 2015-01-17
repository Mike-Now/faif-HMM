
#ifndef FAIF_MLReg_HPP
#define FAIF_MLReg_HPP

#include <string>
#include <map>
#include <memory>
#include <vector>
#include <boost/functional/factory.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/static_assert.hpp>
#include <boost/multi_array.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include "Classifier.hpp"
#include "boostFix.hpp"

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

                    /* typedef std::string DomainId; */
                    typedef boost::multi_array<double, 2> NormalizedExamples;
                    typedef std::unique_ptr< NormalizedExamples> NormalizedExamplesPtr;
                    typedef std::unique_ptr< MLRegTraining> MLRegTrainingPtr;
                    typedef boost::function< MLRegTrainingPtr ()> TrainingFactory;
                private:
                    class Model;
                    class FactoryManager;
                    std::unique_ptr<Model> model;
                    std::unique_ptr<MLRegTraining> trainingImpl;
                    std::string currentTrainingId;
                public:
                    MLReg();
                    MLReg(const Domains& attr_domains, const AttrDomain& category_domains,std::string algorithmId );
                    virtual ~MLReg() { }

                    virtual void reset();
                    virtual void reset(std::string algorithmId);

                    /*template<class Archive>
                    void save(Archive & ar, const unsigned int file_version) const {
                        ar & currentTrainingId;
                        ar & *model;
                        ar & *trainingImpl;
                    }

                    template<class Archive>
                    void load(Archive & ar, const unsigned int ile_version) {

                    }*/

                    template<class Archive>
                    void serialize( Archive &ar, const unsigned int file_version ){
                        //boost::serialization::split_member(ar, *this, file_version);
                        boost::serialization::base_object<Classifier<Val> >(*this);

                        ar & currentTrainingId;
                        ar & model;
                        ar & trainingImpl;
                    }

                    std::string getTrainingId(){return currentTrainingId;}

                    template<class T>
                        static void registerTraining(std::string trainingId);

                    AttrIdd getCategory(const ExampleTest& example) const;

                    Beliefs getCategories(const ExampleTest& example) const;

                    /** \brief train classifier */
                    virtual void train(const ExamplesTrain& e);

                    /** the ostream method */
                    virtual void write(std::ostream& os) const;

                    class MLRegTraining {
                        friend class boost::serialization::access;

                        template<class Archive>
                        void serialize(Archive & ar, const unsigned int version)
                        {

                        }

                        public:
                            virtual void train(NormalizedExamples& examples)=0;
                            virtual ~MLRegTraining(){};
                    };
                    class GISTraining : public MLRegTraining{
                        friend class boost::serialization::access;

                        template<class Archive>
                        void serialize(Archive & ar, const unsigned int version)
                        {
                            boost::serialization::base_object<Classifier<MLRegTraining> >(*this);
                        }
                        public:
                            void train(NormalizedExamples& examples);
                            ~GISTraining(){}
                    };
                private:
                    class FactoryManager{
                        private:
                            typedef typename MLReg<Val>::TrainingFactory TrainingFactory;
                            std::map<std::string,TrainingFactory> trainings;
                        public:
                            FactoryManager();
                            //C++11 threadsafe lazy initialization
                            static FactoryManager& getInstance(){

                                static FactoryManager instance;
                                return instance;

                            }
                            TrainingFactory getFactory(std::string trainingId);
                            template<class T>
                                void registerTraining(std::string trainingId);
                        private:
                            ~FactoryManager(){}
                            FactoryManager(const TrainingFactory&t);
                            FactoryManager& operator=(const TrainingFactory&);

                            friend class boost::serialization::access;

                            template<class Archive>
                            void serialize( Archive &ar, const unsigned int file_version ){
                                /* boost::serialization::split_member(ar, *this, file_version); */
                                ar & trainings;
                            }

                    };
                    class Model{
                        public:
                            typedef int NAttrId;
                            typedef int CategoryId;
                            NormalizedExamplesPtr normalizeExamples(const ExamplesTrain& examples) const;
                            AttrIdd classify(const ExampleTest& testEx);
                            Model(MLReg & parent): parent_(&parent){
                                mapAttributes();
                            }

                            Probability calcProbabilityForExample(const ExampleTest& example, AttrIdd cat_val) const;

                            AttrIdd getCategory(const ExampleTest&) const;

                            Beliefs getCategories(const ExampleTest&) const;
                            //infer
                        private:
                            void mapAttributes();
                            //initial values -> normalized values mapping
                            std::map<AttrIdd, NAttrId> NormMap;
                            //trained params
                            std::vector<double> parameters;
                            MLReg * parent_;
                            //Naive : 643 //TODO
                            /** \brief serialization using boost::serialization */
                            friend class boost::serialization::access;

                            /*template<class Archive>
                            void save(Archive & ar, const unsigned int file_version ) const {
                                ar << boost::serialization::make_nvp("Category", data_ );
                                 ar << boost::serialization::make_nvp("Data", attrData_ );

                            }*/

                            /*template<class Archive>
                            void load(Archive & ar, const unsigned int file_version) {
                                 ar >> boost::serialization::make_nvp("Category", data_ );
                                 typedef std::map<AttrIddSerialize,T> Map;
                                 Map m;
                                 ar >> boost::serialization::make_nvp("Data", m );
                                 attrData_.clear();
                                 for(typename Map::const_iterator ii = m.begin(); ii != m.end(); ++ii) {
                                     //transform from loaded std::pair (with not const key) to stored std::pair is required
                                     attrData_.insert(typename AttrData::value_type(ii->first, ii->second) );
                                 }
                            }*/

                            template<class Archive>
                            void serialize( Archive &ar, const unsigned int file_version ){
                                /* boost::serialization::split_member(ar, *this, file_version); */
                                ar & parameters;
                                ar & NormMap;
                            }
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
                model.reset(new Model(*this));
                currentTrainingId = trainingId;
                this->reset(trainingId);
            }

        template<typename Val>
            void MLReg<Val>::train(const ExamplesTrain& examples) {
                NormalizedExamplesPtr ptr = model->normalizeExamples(examples);
                trainingImpl->train(*ptr);
            };
        /** clear the learned parameters */
        template<typename Val>
            void MLReg<Val>::reset() {
                model.reset(new Model(*this));
                /* this->reset(currentTrainingId); */
            };

        template<typename Val>
            void MLReg<Val>::reset(std::string treningId){
                TrainingFactory factory;
                factory = FactoryManager::getInstance().getFactory(treningId);
                trainingImpl=factory();
                model.reset(new Model(*this));
            };

        template<typename Val>
            typename MLReg<Val>::AttrIdd
            MLReg<Val>::getCategory(const ExampleTest& example) const {
                return model->getCategory(example);
        };

        template<typename Val>
        typename MLReg<Val>::Beliefs
        MLReg<Val>::getCategories(const ExampleTest& example) const {
            return model->getCategories(example);
        }
        template<typename Val>
            template<class T>
            void MLReg<Val>::registerTraining(std::string trainingId){
                FactoryManager::getInstance().template registerTraining<T>(trainingId);
            }
        /** ostream method */
        template<typename Val>
            void MLReg<Val>::write(std::ostream& os) const {
            }

        //////////////////////////////////////////////////////////////////////////////////////////////////
        // class FactoryManager implementation
        //////////////////////////////////////////////////////////////////////////////////////////////////
        template<typename Val>
            MLReg<Val>::FactoryManager::FactoryManager(){
                registerTraining<MLReg<Val>::GISTraining>("GIS");
            }
        template<typename Val>
            typename MLReg<Val>::TrainingFactory
            MLReg<Val>::FactoryManager::
            getFactory(std::string trainingId){
                typename std::map<std::string, TrainingFactory>::iterator it;
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
            void MLReg<Val>::FactoryManager::
            registerTraining(std::string trainingId)
            {
                BOOST_STATIC_ASSERT_MSG(
                        (std::is_base_of<MLRegTraining, T>::value),
                        "A registered training must be a descendant of MLRegTraining."
                        );
                MLReg<Val>::TrainingFactory factory = boost::factory<std::unique_ptr<T>>();
                this->trainings.insert(std::make_pair(trainingId,factory));

            }
        //////////////////////////////////////////////////////////////////////////////////////////////////
        // class Model implementation
        //////////////////////////////////////////////////////////////////////////////////////////////////

        template<typename Val>
            void MLReg<Val>::Model::mapAttributes() {
                //TODO
            }
        template<typename Val>
        typename MLReg<Val>::NormalizedExamplesPtr
        MLReg<Val>::Model::normalizeExamples(const MLReg<Val>::ExamplesTrain& examples)const {
                //TODO
                NormalizedExamplesPtr ptr;
                return ptr;
            }
        template<typename Val>
            typename MLReg<Val>::AttrIdd
            MLReg<Val>::Model::getCategory(const ExampleTest& example) const {
                if( parameters.empty() )
                    return AttrDomain::getUnknownId();

                /* AttrIdd cat_val_max = probabl_.begin()->first; //init not important */
                /* Probability max_prob = -std::numeric_limits<Probability>::max(); */
                //look the categories and find the max prob of category for given example (compares the log of probability)
                /* for(typename InternalProbabilities::const_iterator ii = probabl_.begin(); ii != probabl_.end(); ++ii ) { */
                /*     AttrIdd cat_val = (*ii).first; */
                /*     Probability prob = calcProbabilityForExample(example, cat_val); */
                /*     if( prob > max_prob ) { */
                /*         max_prob = prob; */
                /*         cat_val_max = cat_val; */
                /*     } */
                /* } */
                /* return cat_val_max; */
                return parent_->getCategoryIdd("good");
    }

        template<typename Val>
        typename MLReg<Val>::Beliefs
        MLReg<Val>::Model::getCategories(const ExampleTest& example) const {
            /* return impl_->getCategories(example); */
            Beliefs b;
            return b;
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////
        //class MLReg::GISTraining
        //////////////////////////////////////////////////////////////////////////////////////////////////

        template<typename Val>
        void MLReg<Val>::GISTraining::train(MLReg<Val>::NormalizedExamples& examples){
        }
    }
}
#endif
