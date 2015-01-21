
#ifndef FAIF_MLReg_HPP
#define FAIF_MLReg_HPP
#include <iostream>
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
                    struct Matrix: boost::multi_array<double, 2>{
                        Matrix(int n,int m): boost::multi_array<double,2>(boost::extents[n][m]){
                            std::fill(this->data(),this->data()+this->num_elements(),0);
                        }
                        void print(){
                            std::cout<<"Macierz:"<<std::endl;
                            for(int i=0;i<size1();i++){
                                for(int j=0;j<size2();j++){
                                    std::cout<<(*this)[i][j]<<" ";
                                }
                                std::cout<<std::endl;
                            }
                        }
                        int size1() const {return this->shape()[0];}
                        int size2() const {return this->shape()[1];}
                    };
                    struct Vector: boost::multi_array<double ,1>{
                        Vector(int n): boost::multi_array<double,1>(boost::extents[n]){
                            std::fill(this->data(),this->data()+this->num_elements(),0);
                        }
                        void print(){
                            std::cout<<"Wektor:"<<std::endl;
                            for(int i=0;i<size();i++){
                                std::cout<<(*this)[i]<<" ";
                            }
                            std::cout<<std::endl;

                        }
                        int size(){return this->shape()[0];}
                    };
                    /* typedef boost::multi_array<double, 1> ParamVector; */
                    typedef boost::multi_array<int , 1>IntVec;

                    struct NormExample: Vector{
                        NormExample(int size): Vector(size) {
                            std::fill(this->data(),this->data()+this->num_elements(),0);
                            category=-1;
                        }
                        NCategoryId category;
                        int size() const {return this->shape()[0];}
                        void print(){
                            std::cout<<"Wektor: ";
                            for(int i=0;i<size();i++){
                                std::cout<<(*this)[i]<<" ";
                            }
                            if(category!=-1)
                                std::cout<<"Kat.: "<<category;
                            std::cout<<std::endl;
                        }
                    };

                    typedef std::unique_ptr< NormExample> NormExamplePtr;

                    struct NormExamples: std::vector<NormExample>{
                        NormExamples(int catN,int attrN, int exNum): std::vector<NormExample>(exNum,NormExample(attrN)){
                            categoriesCount=catN;
                        }
                        NormExamples(int attrN,int exNum):std::vector<NormExample>(exNum,NormExample(attrN)){
                            categoriesCount=-1;
                        }
                        void print(){
                            for(typename std::vector<NormExample>::iterator it=this->begin();it!=this->end();it++){
                                it->print();
                            }

                        }
                        int categoriesCount;
                    };
                    typedef std::unique_ptr< NormExamples> NormExamplesPtr;
                    typedef std::unique_ptr< MLRegTraining> MLRegTrainingPtr;
                    typedef boost::function< MLRegTrainingPtr ()> TrainingFactory;
                private:
                    class Model;
                    class FactoryManager;
                    std::unique_ptr<Model> model;
                    std::unique_ptr<MLRegTraining> trainingImpl;
                    std::string currentTrainingId;
                    static Probability calcSoftMax(const NormExample& ex, const NCategoryId nCatId,const Matrix &parameters);
                public:
                    MLReg();
                    MLReg(const Domains& attr_domains, const AttrDomain& category_domains,std::string algorithmId );
                    virtual ~MLReg() { }

                    virtual void reset();
                    virtual void reset(std::string algorithmId);

                    template<class Archive>
                        void save(Archive & ar, const unsigned int file_version) const {
                            boost::serialization::base_object<Classifier<Val> >(*this);

                            ar & currentTrainingId;
                            ar & model;
                            ar & trainingImpl;
                        }

                    template<class Archive>
                        void load(Archive & ar, const unsigned int ile_version) {
                            boost::serialization::base_object<Classifier<Val> >(*this);

                            ar & currentTrainingId;
                            ar & model;
                            ar & trainingImpl;

                            if(model) {
                                model->parent_ = this;
                                model->mapAttributes();
                            }
                        }

                    template<class Archive>
                        void serialize( Archive &ar, const unsigned int file_version ){
                            boost::serialization::split_member(ar, *this, file_version);
                            /*boost::serialization::base_object<Classifier<Val> >(*this);

                              ar & currentTrainingId;
                              ar & model;
                              ar & trainingImpl;*/
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
                        Probability calcSoftMax(NormExample&ex,NCategoryId&c,Matrix&p){
                            return MLReg::calcSoftMax(ex,c,p);
                        }

                        template<class Archive>
                            void serialize(Archive & ar, const unsigned int version)
                            {

                            }

                        public:
                        virtual Matrix* train(NormExamples& examples)=0;
                        virtual ~MLRegTraining(){};
                    };
                    class BGDTraining : public MLRegTraining{
                        friend class boost::serialization::access;

                        template<class Archive>
                            void serialize(Archive & ar, const unsigned int version)
                            {
                                boost::serialization::base_object<Classifier<MLRegTraining> >(*this);
                            }
                        public:
                        Vector calcChange(NormExamples &examples, Matrix& parameters,NCategoryId catId);
                        Matrix* train(NormExamples& examples);
                        ~BGDTraining(){}
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
                            NormExample normalizeTestExample(const ExampleTest& example)const;
                            NormExamplesPtr normalizeExamples(const ExamplesTrain& examples) const;
                            AttrIdd classify(const ExampleTest& testEx);
                            Model() {
                            }
                            Model(MLReg & parent): parent_(&parent){
                                mapAttributes();
                            }

                            AttrIdd getCategory(const ExampleTest&) const;

                            Beliefs getCategories(const ExampleTest&) const;
                            void setParameters(Matrix *trainedParams);
                            //infer
                        private:
                            void mapAttributes();
                            //initial values -> normalized values mapping
                            std::map<AttrIdd, NAttrId> normMap;
                            //trained params
                            std::unique_ptr<Matrix> parameters;

                            //map between internal cat. indices and category ids(attridd)
                            std::map<NCategoryId, AttrIdd> catMap;
                            std::map<AttrIdd,NCategoryId> revCatMap;
                            MLReg * parent_;

                            /** \brief serialization using boost::serialization */
                            friend class boost::serialization::access;
                            friend class MLReg;

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
                                    ar & catMap;
                                    ar & revCatMap;
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
        template <typename Val>
            Probability
            MLReg<Val>::calcSoftMax(const NormExample& ex, const NCategoryId nCatId,const Matrix &parameters)
            {

                double power=0;
                for(int j=0;j<ex.size();j++){
                    double beta = parameters[nCatId][j];
                    double attrVal = ex[j];
                    power+=beta*attrVal;
                }
                double numerator = std::exp(power);

                double denominator= 0;
                for (int c=0;c<parameters.size1();c++){

                        double power=0;
                        for(int j=0;j<ex.size();j++){

                            double beta = parameters[c][j];
                            double attrVal = ex[j];
                            power+=beta*attrVal;
                        }

                        denominator+=std::exp(power);
                }
                return numerator/denominator;
            }

        template<typename Val>
            void MLReg<Val>::train(const ExamplesTrain& examples) {
                NormExamplesPtr ptr = model->normalizeExamples(examples);
                Matrix * params = trainingImpl->train(*ptr);
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
                registerTraining<MLReg<Val>::BGDTraining>("BGD");
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
            typename MLReg<Val>::NormExamplesPtr
            MLReg<Val>::Model::normalizeExamples(const MLReg<Val>::ExamplesTrain& examples)const {
                /* bool isNominal=(typeid(typename AttrDomain::ValueTag)==typeid(faif::nominal_tag)); */
                int nattrNum = normMap.size();
                int ncatNum = catMap.size();
                int exNum = examples.size();
                NormExamplesPtr normExamples(new NormExamples(ncatNum,nattrNum,exNum));

                typename ExamplesTrain::const_iterator exIt;
                int ii=0;
                for( exIt=examples.begin();exIt!=examples.end();exIt++){

                    const ExampleTrain &ex = *exIt;
                    AttrIdd catVal = ex.getFeature();
                    NCategoryId nCatId = revCatMap.find(catVal)->second;
                    NormExample nEx(nattrNum);
                    nEx.category=nCatId;
                    for(typename ExampleTrain::const_iterator i = ex.begin();i!=ex.end();i++)
                    {
                        AttrIdd trnValue = *i;
                        NAttrId nAttrId = normMap.find(trnValue)->second;
                        nEx[nAttrId]=1.0;
                    }
                    (*normExamples)[ii] = nEx;
                    ii++;
                }
                return normExamples;
            }
        template <typename Val>
            typename MLReg<Val>::NormExample
            MLReg<Val>::Model::normalizeTestExample(const ExampleTest& example)const{
                /* bool isNominal=(typeid(typename AttrDomain::ValueTag)==typeid(faif::nominal_tag)); */
                int vecSize=parameters->size2();
                NormExample ex=NormExample(vecSize);
                for(typename ExampleTest::const_iterator ii=example.begin();ii!=example.end();ii++)
                {
                    NAttrId nAttrId= normMap.find(*ii)->second;
                    ex[nAttrId]=1.0;
                }
                return ex;
            }
        template<typename Val>
            typename MLReg<Val>::AttrIdd
            MLReg<Val>::Model::getCategory(const ExampleTest& example) const {
                Probability maxProb=0;
                NCategoryId bestCatId;
                NormExample ex= normalizeTestExample(example);
                for(int i=0;i<parameters->size1();i++){
                    NCategoryId nCatId= i;
                    Probability prob = calcSoftMax(ex,nCatId,*parameters);
                    if(prob>maxProb){
                        maxProb=prob;
                        bestCatId = nCatId;
                    }

                }

                AttrIdd rCat = catMap.find(bestCatId)->second;
                return rCat;
            }

        template<typename Val>
            typename MLReg<Val>::Beliefs
            MLReg<Val>::Model::getCategories(const ExampleTest& example) const {
                /* return impl_->getCategories(example); */
                //TODO
                Beliefs b;
                return b;
            }
        template<typename Val>
            void MLReg<Val>::Model::setParameters(Matrix * trainParams){
                parameters.reset(trainParams);
            }

        //////////////////////////////////////////////////////////////////////////////////////////////////
        //class MLReg::BGDTraining
        //////////////////////////////////////////////////////////////////////////////////////////////////

        template<typename Val>
            typename MLReg<Val>::Matrix *MLReg<Val>::BGDTraining::train(MLReg<Val>::NormExamples& examples){
                int catN = examples.categoriesCount;
                int attrN = examples[0].size();

                /* examples.print(); */
                Matrix * trainedParams = new Matrix(catN,attrN);
                Matrix & params=*trainedParams;
                /* for(int i=0;i<params.size2();i++) */
                /* { */
                /*     params[0][i]=1.0; */
                /* } */

                double learningRate=1;
                int maxIter=100;
                int iter=0;
                /* double tol=1e-5; */
                bool converged=false;
                while(iter<maxIter && !converged){
                    //iterate for every category
                    for(NCategoryId i=0;i<catN;i++){
                        Vector iterVec = calcChange(examples,params,i);
                        //iterate for every attr weight
                        /* itervec.print(); */
                        double hDelta=0;
                        for(NAttrId j=0;j<attrN;j++){
                            params[i][j]-=learningRate*iterVec[j];
                            if(abs(iterVec[j])>hDelta) hDelta=iterVec[j];
                        }
                        /* if(hDelta<=tol) converged=true; */
                    }
                    iter++;

                }
                /* std::cout<<"ITER"<<iter<<std::endl; */
                /* params.print(); */
                return trainedParams;
            }
        template<typename Val>
            typename MLReg<Val>::Vector
            MLReg<Val>::BGDTraining::calcChange(MLReg<Val>::NormExamples& examples,Matrix& parameters, NCategoryId catId)
            {
                int attrNum = examples[0].size();
                /* int catNum = examples.categoriesCount; */
                int exNum = examples.size();
                Vector delta(attrNum);
                for(typename NormExamples::iterator it=examples.begin();it!=examples.end();it++){
                    double indicatorVal=0;
                    NCategoryId iCatId = it->category;
                    if(iCatId == catId) indicatorVal=1.0;
                    Probability softMaxVal = calcSoftMax(*it,iCatId,parameters);
                    double innerVal=indicatorVal-softMaxVal;
                    for(int i=0;i<delta.size();i++)
                    {
                        delta[i]+=(*it)[i]*innerVal;
                    }
                }
                /* delta.print(); */
                for(int i=0;i<delta.size();i++){
                    /* delta[i]*=-1/exNum; */
                    delta[i]=delta[i]*-1/exNum;
                }
                return delta;

            }

    }
}
#endif
