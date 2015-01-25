
#ifndef FAIF_MLReg_HPP
#define FAIF_MLReg_HPP
#include <iostream>
#include <string>
#include <map>
#include <memory>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <boost/functional/factory.hpp>
#include <boost/function.hpp>
#include <boost/static_assert.hpp>
#include <boost/multi_array.hpp>
#include <boost/optional.hpp>
#include <boost/any.hpp>
/* #include <boost/serialization/split_member.hpp> */
/* #include <boost/serialization/base_object.hpp> */
/* #include <boost/serialization/nvp.hpp> */
/* #include <boost/serialization/map.hpp> */
/* #include <boost/serialization/vector.hpp> */
#include "Classifier.hpp"
#include "MachineLearningExceptions.hpp"

namespace faif {
    namespace ml {
        /** Multinomial Logistic Regression Classifier
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

                    /**
                     * a container for training parameters.
                     *
                     * the convenience class to set dynamic parameters.
                     */
                    struct TrainingParameters : std::map<std::string,boost::any>{

                        template<class T>
                            T get(const std::string key)const{
                                iterator it = this->find(key);
                                if(it==this->end())
                                    throw std::runtime_error(std::string("No such training parameter: ")+key);
                                else
                                    return boost::any_cast<T>(it->second);
                            }
                        bool exists(const std::string key){
                            iterator it = this->find(key);
                            if(it==this->end())
                                return false;
                            else
                                return true;
                        }
                        private:
                        typedef std::map<std::string,boost::any>::const_iterator iterator;

                    };

                    //forward declaration of an abstract training class
                    class MLRegTraining;

                    //internal indices for categories(classes) and attributes(features).
                    typedef int IAttrId;
                    typedef int ICategoryId;


                    /**
                     * a dynamic matrix for values of 'double' type.
                     */
                    struct Matrix: boost::multi_array<double, 2>{
                        Matrix(int n,int m): boost::multi_array<double,2>(boost::extents[n][m]){
                            reset();
                        }

                        void print() const{
                            std::cout<<"Matrix:"<<std::endl;
                            for(int i=0;i<size1();i++){
                                for(int j=0;j<size2();j++){
                                    std::cout<<(*this)[i][j]<<" ";
                                }
                                std::cout<<std::endl;
                            }
                        }
                        double absMax(){
                            double absMax=0.0;
                            for(int i=0;i<size1();i++){
                                for(int j=0;j<size2();j++){
                                    double t= fabs((*this)[i][j]);
                                    if(t>absMax)
                                        absMax=t;
                                }
                            }
                            return absMax;
                        }

                        int size1() const {return this->shape()[0];}
                        int size2() const {return this->shape()[1];}

                        /**
                         * set the value of every cell
                         */
                        void reset(double r=0.0)
                        {
                            std::fill(this->data(),this->data()+this->num_elements(),r);
                        }

                        /**
                         * assign operator overload
                         *
                         * uses boost's elementwise copying to prevent
                         * launching of relatively expensive copy-constructor*/
                        Matrix& operator =(const Matrix &m){
                            boost::multi_array<double, 2>::operator = (m);
                            return *this;
                        }
                    };


                    /**
                     * essentially a one dimensional matrix.
                     */
                    struct Vector: boost::multi_array<double ,1>{
                        Vector(int n): boost::multi_array<double,1>(boost::extents[n]){
                            reset();
                        }
                        void print() const{
                            std::cout<<"Vector:"<<std::endl;
                            for(int i=0;i<size();i++){
                                std::cout<<(*this)[i]<<" ";
                            }
                            std::cout<<std::endl;

                        }
                        void reset(double r=0.0)
                        {
                            std::fill(this->data(),this->data()+this->num_elements(),r);
                        }
                        Vector& operator =(const Vector &v){
                            boost::multi_array<double, 1>::operator = (v);
                            return *this;
                        }

                        int size()const{return this->shape()[0];}
                    };



                    /**
                     * the internal representation of a training sample.
                     *
                     * inherits from the Vector struct.
                     */
                    struct IExample: Vector{
                        IExample(int size): Vector(size) {
                            category=-1;
                        }
                        ICategoryId category;
                        int size() const {return this->shape()[0];}
                        void print(){
                            Vector::print();
                            if(category!=-1)
                                std::cout<<"Cat.: "<<category;
                            std::cout<<std::endl;
                        }
                    };

                    typedef std::unique_ptr< IExample> IExamplePtr;

                    /**
                     * a vector of training samples.
                     */
                    struct IExamples: std::vector<IExample>{
                        IExamples(int catN,int attrN, int exNum): std::vector<IExample>(exNum,IExample(attrN)){
                            categoriesCount=catN;
                        }
                        IExamples(int attrN,int exNum):std::vector<IExample>(exNum,IExample(attrN)){
                            categoriesCount=-1;
                        }
                        void print(){
                            for(typename IExample::iterator it=this->begin();it!=this->end();it++){
                                it->print();
                            }
                        }

                        int categoriesCount;
                    };

                    typedef std::unique_ptr< IExamples> IExamplesPtr;
                    typedef std::unique_ptr< MLRegTraining> MLRegTrainingPtr;

                    //a generic factory for a training algorithm
                    typedef boost::function< MLRegTrainingPtr ()> TrainingFactory;
                private:

                    //stores trained parameters and calculates probability
                    class Model;

                    //stores different training algorithms
                    class FactoryManager;

                    //the current model
                    std::unique_ptr<Model> model;

                    //the current training algorithm
                    std::unique_ptr<MLRegTraining> trainingImpl;

                    //the id of the current training algorithm
                    std::string currentTrainingId;

                    //a user-defined training settings
                    TrainingParameters trainingParameters;

                public:
                    MLReg();
                    MLReg(const Domains& attr_domains, const AttrDomain& category_domains,std::string algorithmId );
                    virtual ~MLReg() { }

                    virtual void reset();
                    virtual void reset(std::string algorithmId);

                    static Probability calcSoftMax(const IExample& ex, const ICategoryId nCatId,const Matrix &parameters,double constant=0.0);

                    void setTrainingParameters(TrainingParameters tParams){
                        trainingParameters = tParams;
                        trainingImpl->setParameters(trainingParameters);
                    }

                    std::string getTrainingId(){return currentTrainingId;}

                    double validateModel(const ExamplesTrain& e);
                    template<class T>
                        static void registerTraining(std::string trainingId);

                    AttrIdd getCategory(const ExampleTest& example) const;

                    Beliefs getCategories(const ExampleTest& example) const;

                    /** \brief train classifier */
                    virtual void train(const ExamplesTrain& e);

                    /** the ostream method */
                    virtual void write(std::ostream& os) const;

                    class MLRegTraining {
                        Probability calcSoftMax(IExample&ex,ICategoryId&c,Matrix&p,double cons){
                            return MLReg::calcSoftMax(ex,c,p,cons);
                        }

                        public:
                        virtual void setParameters(TrainingParameters &p)=0;
                        virtual Matrix* train(IExamples& examples)=0;
                        virtual ~MLRegTraining(){};
                    };
                    class BGDTraining : public MLRegTraining{
                        friend class boost::serialization::access;

                        public:
                        double learningRate;
                        int totalIterations;
                        BGDTraining():learningRate(1.0),totalIterations(1000),constant(0.0){}
                        void setParameters(TrainingParameters &p);
                        double calcCost(MLReg<Val>::IExamples& example,Matrix& parameters);
                        Vector calcGrad(IExamples &examples, Matrix& parameters,ICategoryId catId);
                        Matrix* train(IExamples& examples);
                        ~BGDTraining(){}
                        private:
                        double constant;
                    };
                private:

                    /**
                     * The class for storing training algorithms.
                     *
                     */
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

                    };
                    class Model{
                        public:
                            IExample mapTestExample(const ExampleTest& example)const;
                            IExamplesPtr mapExamples(const ExamplesTrain& examples) const;
                            AttrIdd classify(const ExampleTest& testEx);
                            Model(MLReg & parent): parent_(&parent){
                                mapAttributes();
                            }

                            AttrIdd getCategory(const ExampleTest&) const;

                            Beliefs getCategories(const ExampleTest&) const;
                            void setParameters(Matrix *trainedParams);

                        private:
                            Model();
                            void mapAttributes();

                            std::map<AttrIdd, IAttrId> attrMap;

                            //trained params
                            std::unique_ptr<Matrix> parameters;


                            //map between internal cat. indices and category ids(attridd)
                            std::map<ICategoryId, AttrIdd> catMap;

                            //reverse of the map above
                            std::map<AttrIdd,ICategoryId> revCatMap;

                            //cached constant to prevent overflow
                            double constant;

                            MLReg * parent_;

                            friend class MLReg;

                    };
            }; //class MLReg

        //////////////////////////////////////////////////////////////////////////////////////////////////
        // class MLReg implementation
        //////////////////////////////////////////////////////////////////////////////////////////////////

        /**
         * the empty classifier constructor
         *
         * used for unmarshalling.
         */
        template<typename Val>
            MLReg<Val>::MLReg() : Classifier<Val>()
        {
        }

        /**
         * initializes model and a training algorithm
         */
        template<typename Val>
            MLReg<Val>::MLReg(const Domains& attr_domains, const AttrDomain& category_domain,std::string trainingId)
            : Classifier<Val>(attr_domains, category_domain)
            {
                model.reset(new Model(*this));
                currentTrainingId = trainingId;
                this->reset(trainingId);
            }

        /**
         * implementation of softmax for regression
         *
         * @param ex a sample
         * @param nCatId a category for which to calculate probability
         * @param parameters a set of trained parameters
         * @constant an offset to prevent overflow
         */
        template <typename Val>
            Probability
            MLReg<Val>::calcSoftMax(const IExample& ex, const ICategoryId nCatId,const Matrix &parameters,double constant)
            {
                double power=0;
                for(int j=0;j<ex.size();j++){
                    double attrVal = ex[j];

                    //skips an unnecessary iteration
                    if(attrVal==0.0) continue;

                    double beta = parameters[nCatId][j];
                    power=power+beta*attrVal;
                }
                double numerator = std::exp(power-constant);

                double denominator= 0;
                for (int c=0;c<parameters.size1();c++){

                    double power=0;
                    for(int j=0;j<ex.size();j++){

                        double attrVal = ex[j];

                        if(attrVal==0.0)continue;

                        double beta = parameters[c][j];
                        power=power+beta*attrVal;
                    }

                    denominator+=std::exp(power-constant);
                }
                return numerator/denominator;
            }

        /**
         * fits model based on offline samples
         */
        template<typename Val>
            void MLReg<Val>::train(const ExamplesTrain& examples) {
                IExamplesPtr ptr = model->mapExamples(examples);
                Matrix * params = trainingImpl->train(*ptr);
                model->setParameters(params);
            };

        /**
         * clears regression parameters
         */
        template<typename Val>
            void MLReg<Val>::reset() {
                model.reset(new Model(*this));
            };

        /**
         * clears regression parameters
         *
         * sets training algorithm based on a given trainingId
         */
        template<typename Val>
            void MLReg<Val>::reset(std::string trainingId){
                TrainingFactory factory;
                factory = FactoryManager::getInstance().getFactory(trainingId);
                trainingImpl=factory();
                trainingImpl->setParameters(trainingParameters);
                model.reset(new Model(*this));
            };

        /**
         * returns the best category based on a trained model
         */
        template<typename Val>
            typename MLReg<Val>::AttrIdd
            MLReg<Val>::getCategory(const ExampleTest& example) const {
                return model->getCategory(example);
            };

        /**
         * returns a score (softmax probability) for an every class
         */
        template<typename Val>
            typename MLReg<Val>::Beliefs
            MLReg<Val>::getCategories(const ExampleTest& example) const {
                return model->getCategories(example);
            }

        /**
         * registers a training to the FactoryManager
         *
         * templated for convenience - avoids explicit casting
         */
        template<typename Val>
            template<class T>
            void MLReg<Val>::registerTraining(std::string trainingId){
                FactoryManager::getInstance().template registerTraining<T>(trainingId);
            }

        /**
         * prints model parameters to stream
         */
        template<typename Val>
            void MLReg<Val>::write(std::ostream& os) const {
            }

        //////////////////////////////////////////////////////////////////////////////////////////////////
        // class FactoryManager implementation
        //////////////////////////////////////////////////////////////////////////////////////////////////

        /**
         * registers implemented training algorithms
         */
        template<typename Val>
            MLReg<Val>::FactoryManager::FactoryManager(){
                registerTraining<MLReg<Val>::BGDTraining>("BGD");
            }

        /**
         * creates and returns a training algorithm.
         */
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
                    throw MissingTrainingException(trainingId);
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
                IAttrId nAttrIdd=0;
                for(typename Domains::const_iterator jj = attribs.begin(); jj!= attribs.end();++jj){
                    const AttrDomain& attr = *jj;
                    /* int size = attr.getSize(); */
                    for(typename AttrDomain::const_iterator kk = attr.begin(); kk!= attr.end(); ++kk){
                        /* if(size>1 && std::next(kk) == attr.end()) break;//TODO */
                        AttrIdd val = AttrDomain::getValueId(kk);
                        attrMap.insert(std::make_pair(val,nAttrIdd));
                        nAttrIdd+=1;
                    }
                }

                ICategoryId nCatId = 0;
                const AttrDomain& category = this->parent_->getCategoryDomain();
                for(typename AttrDomain::const_iterator ii = category.begin(); ii!=category.end();++ii){
                    AttrIdd catId = AttrDomain::getValueId(ii);
                    catMap.insert(std::make_pair(nCatId,catId));
                    revCatMap.insert(std::make_pair(catId,nCatId));
                    nCatId++;
                }

            }
        template<typename Val>
            typename MLReg<Val>::IExamplesPtr
            MLReg<Val>::Model::mapExamples(const MLReg<Val>::ExamplesTrain& examples)const {
                //possible ugly solution for dispatching types based on nested type ValueNominal<T>
                /* bool isNominal=(typeid(typename AttrDomain::ValueTag)==typeid(faif::nominal_tag)); */
                int nattrNum = attrMap.size();
                int ncatNum = catMap.size();
                int exNum = examples.size();
                IExamplesPtr normExamples(new IExamples(ncatNum,nattrNum,exNum));

                typename ExamplesTrain::const_iterator exIt;
                int ii=0;
                for( exIt=examples.begin();exIt!=examples.end();exIt++){

                    const ExampleTrain &ex = *exIt;
                    AttrIdd catVal = ex.getFeature();
                    ICategoryId nCatId = revCatMap.find(catVal)->second;
                    IExample nEx(nattrNum);
                    nEx.category=nCatId;
                    typename std::map<AttrIdd,IAttrId>::const_iterator mapIt;
                    for(typename ExampleTrain::const_iterator i = ex.begin();i!=ex.end();i++)
                    {
                        AttrIdd trnValue = *i;
                        mapIt = attrMap.find(trnValue);
                        if(mapIt==attrMap.end()) //kth value is missing in k-1 dummy coding
                        {
                            continue;
                        }
                        IAttrId nAttrId = mapIt->second;
                        nEx[nAttrId]=1.0;
                    }
                    (*normExamples)[ii] = nEx;
                    ii++;
                }
                return normExamples;
            }
        template <typename Val>
            typename MLReg<Val>::IExample
            MLReg<Val>::Model::mapTestExample(const ExampleTest& example)const{
                //possible ugly solution for dispatching types based on nested type ValueNominal<T>
                /* bool isNominal=(typeid(typename AttrDomain::ValueTag)==typeid(faif::nominal_tag)); */
                int vecSize=parameters->size2();
                IExample ex=IExample(vecSize);
                typename std::map<AttrIdd,IAttrId>::const_iterator mapIt;
                for(typename ExampleTest::const_iterator ii=example.begin();ii!=example.end();ii++)
                {
                    mapIt = attrMap.find(*ii);

                    //skip null k-th value , when k-1 dummy coding is active
                    if(mapIt==attrMap.end())
                        continue;

                    IAttrId nAttrId= mapIt->second;
                    ex[nAttrId]=1.0;
                }
                return ex;
            }
        template<typename Val>
            typename MLReg<Val>::AttrIdd
            MLReg<Val>::Model::getCategory(const ExampleTest& example) const {
                Probability maxProb=0;
                ICategoryId bestCatId;
                IExample ex= mapTestExample(example);
                for(int i=0;i<parameters->size1();i++){
                    ICategoryId nCatId= i;
                    Probability prob = calcSoftMax(ex,nCatId,*parameters, this->constant);
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
                Beliefs b;
                IExample ex = mapTestExample(example);
                for(int i=0;i<parameters->size1();i++){
                    ICategoryId nCatId= i;
                    Probability prob = calcSoftMax(ex,nCatId,*parameters,this->constant);

                    AttrIdd catVal = catMap.find(nCatId)->second;
                    b.push_back(typename Beliefs::value_type(catVal,prob));
                }
                std::sort(b.begin(),b.end());
                return b;
            }
        template<typename Val>
            void MLReg<Val>::Model::setParameters(Matrix * trainParams){
                parameters.reset(trainParams);
                this->constant = parameters->absMax();
            }

        //////////////////////////////////////////////////////////////////////////////////////////////////
        //class MLReg::BGDTraining
        //////////////////////////////////////////////////////////////////////////////////////////////////

        template<typename Val>
            typename MLReg<Val>::Matrix *MLReg<Val>::BGDTraining::train(MLReg<Val>::IExamples& examples){
                int catN = examples.categoriesCount;
                int attrN = examples[0].size();

                Matrix * trainedParams = new Matrix(catN,attrN);
                Matrix & params=*trainedParams;

                int iter=0;
                Matrix newParams(catN,attrN);

                while(iter<totalIterations){
                    this->constant = params.absMax();

                    for(ICategoryId i=0;i<catN;i++){
                        Vector errorVec = calcGrad(examples,params,i);

                        for(IAttrId j=0;j<attrN;j++){
                            newParams[i][j]=newParams[i][j]-(learningRate*errorVec[j]);
                        }

                    }
                    params=newParams;
                    iter++;

                }

                return trainedParams;
            }
        template<typename Val>
            typename MLReg<Val>::Vector
            MLReg<Val>::BGDTraining::calcGrad(MLReg<Val>::IExamples& examples,Matrix& parameters, ICategoryId catId)
            {
                int attrNum = examples[0].size();

                int exNum = examples.size();
                Vector grad(attrNum);
                for(typename IExamples::iterator it=examples.begin();it!=examples.end();it++){
                    double indicatorVal=0.0;
                    ICategoryId iCatId = it->category;
                    if(iCatId == catId) indicatorVal=1.0;

                    Probability prob = calcSoftMax(*it,catId,parameters,this->constant);
                    double multiplier=(indicatorVal-prob)/-exNum;

                    for(int i=0;i<grad.size();i++)
                    {
                        if((*it)[i] == 0.0) continue;
                        grad[i]=grad[i]+(*it)[i]*multiplier;
                    }
                }

                return grad;

            }
        template<typename Val>
            double MLReg<Val>::BGDTraining::calcCost(MLReg<Val>::IExamples& examples,Matrix& parameters){
                double multiplier = -1/examples.size();
                double cost =0.0;
                for(int i=0;i<examples.size();i++){
                    for(ICategoryId j=0;j<parameters.size1();j++){
                        if(j==examples[i].category){
                            Probability prob = calcSoftMax(examples[i],j,parameters,this->constant);
                            cost+=std::log(prob)*multiplier;

                        }
                    }
                }
                return cost;

            }
        template<typename Val>
            void MLReg<Val>::BGDTraining::setParameters(MLReg<Val>::TrainingParameters &p){
                if(p.exists("totalIterations")){
                    int t = p.template get<int>("totalIterations");
                    this->totalIterations=t;}
                if(p.exists("learningRate")){
                    double d = p.template get<double>("learningRate");
                    this->learningRate= d;}
            }

    }
}
#endif
