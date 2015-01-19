
#ifndef FAIF_MLReg_HPP
#define FAIF_MLReg_HPP

#include <string>
#include <map>
#include <memory>
#include <vector>
#include <cmath>
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
                    typedef int NAttrId;
                    typedef int NCategoryId;
                    typedef boost::multi_array<double, 2> ParamMatrix;
                    typedef boost::multi_array<int , 1>IntVec;

                    struct NormTrainingExamples{
                        NormTrainingExamples(int catN, int attrN, int exNum){
                            params.reset(new ParamMatrix(boost::extents[exNum][attrN]));
                            std::fill(params->data(),params->data()+params->num_elements(),0);
                            catVector.reset(new IntVec(boost::extents[exNum]));
                            std::fill(catVector->data(),catVector->data()+catVector->num_elements(),0);
                            categoriesCount=catN;
                        }
                        void print(){

                            for(int i=0;i<catVector->shape()[0];i++)
                            {
                                std::cout<<"Wektor: ";
                                for(int j=0;j<params->shape()[1];j++){
                                    std::cout<<(*params)[i][j]<<" ";
                                }
                                std::cout<<"Kat. : "<<(*catVector)[i]<<std::endl;
                            }
                        }
                        int categoriesCount;
                        std::unique_ptr<ParamMatrix> params;
                        std::unique_ptr<IntVec>catVector;
                    };
                    struct NormTestExample: boost::multi_array<double, 1>{
                        NormTestExample(int size): boost::multi_array<double, 1>(boost::extents[size]){
                            std::fill(this->data(),this->data()+this->num_elements(),0);
                        }
                        void print(){
                            std::cout<<"Wektor: ";
                            for(int i=0;i<this->shape()[0];i++){
                                std::cout<<(*this)[i]<<" ";
                            }
                            std::cout<<std::endl;
                        }
                    };
                    typedef std::unique_ptr< NormTrainingExamples> NormTrainingExamplesPtr;
                    typedef std::unique_ptr< NormTestExample> NormTestExamplePtr;
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
                        virtual ParamMatrix* train(NormTrainingExamples& examples)=0;
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
                        ParamMatrix* train(NormTrainingExamples& examples);
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
                            NormTestExamplePtr normalizeTestExample(const ExampleTest& example)const;
                            NormTrainingExamplesPtr normalizeExamples(const ExamplesTrain& examples) const;
                            AttrIdd classify(const ExampleTest& testEx);
                            Model(MLReg & parent): parent_(&parent){
                                mapAttributes();
                            }

                            Probability calcProbabilityForExample(const NormTestExample& ex, const NCategoryId nCatId) const;

                            AttrIdd getCategory(const ExampleTest&) const;

                            Beliefs getCategories(const ExampleTest&) const;
                            void setParameters(ParamMatrix *trainedParams);
                            //infer
                        private:
                            void mapAttributes();
                            //initial values -> normalized values mapping
                            std::map<AttrIdd, NAttrId> normMap;
                            //trained params
                            std::unique_ptr<ParamMatrix> parameters;

                            //map between internal cat. indices and category ids(attridd)
                            std::map<NCategoryId, AttrIdd> catMap;
                            std::map<AttrIdd,NCategoryId> revCatMap;
                            MLReg * parent_;

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
                                    ar & normMap;
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
                NormTrainingExamplesPtr ptr = model->normalizeExamples(examples);
                ParamMatrix * params = trainingImpl->train(*ptr);
                model->setParameters(params);
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
            MLReg<Val>::FactoryManager::getFactory(std::string trainingId){
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
                const Domains& attribs = this->parent_->getAttrDomains();
                NAttrId nAttrIdd=0;
                for(typename Domains::const_iterator jj = attribs.begin(); jj!= attribs.end();++jj){
                    const AttrDomain& attr = *jj;
                    for(typename AttrDomain::const_iterator kk = attr.begin(); kk!= attr.end(); ++kk){
                        AttrIdd val = AttrDomain::getValueId(kk);
                        normMap.insert(std::make_pair(val,nAttrIdd));
                        nAttrIdd+=1;
                    }
                }

                NCategoryId nCatId = 0;
                const AttrDomain& category = this->parent_->getCategoryDomain();
                for(typename AttrDomain::const_iterator ii = category.begin(); ii!=category.end();++ii){
                    AttrIdd catId = AttrDomain::getValueId(ii);
                    catMap.insert(std::make_pair(nCatId,catId));
                    revCatMap.insert(std::make_pair(catId,nCatId));
                    nCatId++;
                }

            }
        template<typename Val>
            typename MLReg<Val>::NormTrainingExamplesPtr
            MLReg<Val>::Model::normalizeExamples(const MLReg<Val>::ExamplesTrain& examples)const {
                /* bool isNominal=(typeid(typename AttrDomain::ValueTag)==typeid(faif::nominal_tag)); */
                int nattrNum = normMap.size();
                int ncatNum = catMap.size();
                int exNum = examples.size();
                NormTrainingExamplesPtr examplesPtr(new NormTrainingExamples(ncatNum,nattrNum,exNum));

                typename ExamplesTrain::const_iterator exIt;
                int ii=0;
                for( exIt=examples.begin();exIt!=examples.end();exIt++){

                    const ExampleTrain &ex = *exIt;
                    AttrIdd catVal = ex.getFeature();
                    NCategoryId nCatId = revCatMap.find(catVal)->second;
                    (*examplesPtr->catVector)[ii]=nCatId;
                    for(typename ExampleTrain::const_iterator i = ex.begin();i!=ex.end();i++)
                    {
                        AttrIdd trnValue = *i;
                        NAttrId nValId = normMap.find(trnValue)->second;
                        (*examplesPtr->params)[ii][nValId]=1.0;
                    }
                    ii++;
                }
                return examplesPtr;
            }
        template <typename Val>
            typename MLReg<Val>::NormTestExamplePtr
            MLReg<Val>::Model::normalizeTestExample(const ExampleTest& example)const{
                /* bool isNominal=(typeid(typename AttrDomain::ValueTag)==typeid(faif::nominal_tag)); */
                int vecSize=parameters->shape()[1];
                NormTestExamplePtr exPtr = NormTestExamplePtr(new NormTestExample(vecSize));
                for(typename ExampleTest::const_iterator ii=example.begin();ii!=example.end();ii++)
                {
                    NAttrId nAttrId= normMap.find(*ii)->second;
                    (*exPtr)[nAttrId]=1.0;
                }
                return exPtr;
            }
        template <typename Val>
            Probability
            MLReg<Val>::Model::calcProbabilityForExample(const NormTestExample& ex, const NCategoryId nCatId) const
            {

                double denominator= 0;
                for(int i=0;i<parameters->shape()[0];i++){
                    double power=0;
                    for(int j=0;j<ex.shape()[0];j++){
                        double beta = (*parameters)[i][j];
                        double attrVal = ex[j];
                        power+=beta*attrVal;
                    }

                    denominator+=std::exp(power);
                }
                double power=0;
                for(int j=0;j<ex.shape()[0];j++){
                    double beta = (*parameters)[nCatId][j];
                    double attrVal = ex[j];
                    power+=beta*attrVal;
                }
                double numerator = std::exp(power);
                return numerator/denominator;
            }
        template<typename Val>
            typename MLReg<Val>::AttrIdd
            MLReg<Val>::Model::getCategory(const ExampleTest& example) const {
                Probability maxProb=0;
                NCategoryId bestCatId;
                NormTestExamplePtr exPtr = normalizeTestExample(example);
                for(int i=0;i<parameters->shape()[0];i++){
                    NCategoryId nCatId= i;
                    Probability prob = calcProbabilityForExample(*exPtr,nCatId);
                    if(prob>maxProb){
                        maxProb=prob;
                        bestCatId = nCatId;
                    }

                }
                bestCatId=0;
                AttrIdd rCat = catMap.find(bestCatId)->second;
                return rCat;
                /* if( parameters.empty() ) */
                /*     return AttrDomain::getUnknownId(); */

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
                /* NCategoryId catId = -999; */
                /* if((*parameters)[0][1]==1.0) */
                /*     catId=0; */
                /* AttrIdd rCat = catMap.find(catId)->second; */
                /* return rCat; */
                /* return parent_->getNCategoryIdd("good"); */
            }

        template<typename Val>
            typename MLReg<Val>::Beliefs
            MLReg<Val>::Model::getCategories(const ExampleTest& example) const {
                /* return impl_->getCategories(example); */
                Beliefs b;
                return b;
            }
        template<typename Val>
            void MLReg<Val>::Model::setParameters(ParamMatrix * trainParams){
                parameters.reset(trainParams);
            }

        //////////////////////////////////////////////////////////////////////////////////////////////////
        //class MLReg::GISTraining
        //////////////////////////////////////////////////////////////////////////////////////////////////

        template<typename Val>
            typename MLReg<Val>::ParamMatrix *MLReg<Val>::GISTraining::train(MLReg<Val>::NormTrainingExamples& examples){
                int catN = examples.categoriesCount;
                int attrN = examples.params->shape()[1];
                /* std::cout<<attrN<<" "<<catN<<std::endl; */
                examples.print();
                ParamMatrix * trainedParams = new ParamMatrix(boost::extents[catN][attrN]);
                ParamMatrix & params_=*trainedParams;
                for(int i=0;i<params_.shape()[1];i++)
                {
                    params_[0][i]=1.0;
                }
                return trainedParams;
            }
    }
}
#endif
