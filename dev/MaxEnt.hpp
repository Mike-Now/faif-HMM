#ifndef FAIF_MAXENT_HPP
#define FAIF_MAXENT_HPP

#include <Classifier.hpp>

namespace faif {
	namespace ml {

		template<typename Val>
		class MaxEnt : public Classifier<Val> {
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

		public:
			MaxEnt();
			MaxEnt(const Domains& attr_domains, const AttrDomain& category_domains, unique_ptr<MaxEntTraining> trainer);
			virtual ~MaxEnt() { }

			virtual void reset();

		private:
			unique_ptr<MaxEntTraining> impl_;

			//interface
			class MaxEntTraining {
			public:

			};

			class MaxEntClassify : public MaxEntTraining {

			};
		};
	}
}

#endif
