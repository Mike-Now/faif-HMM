/**
   plik zawiera test wszystkich klas
   defines 'main' for all cpp unit test
**/

#if defined(_MSC_VER) && (_MSC_VER >= 1400)
//msvc9.0 generuje smieci dla biblioteki boost::unit_test_framework
#pragma warning(disable:4275)
#endif

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>
