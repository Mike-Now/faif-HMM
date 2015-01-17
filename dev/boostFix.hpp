namespace boost { 
    namespace serialization {

        template<class Archive, class T>
        inline void save(Archive & ar, const std::unique_ptr< T > &t, const unsigned int /*file_version*/){
            // only the raw pointer has to be saved
            const T * const base_pointer = t.get();
            ar & BOOST_SERIALIZATION_NVP(base_pointer);
        }
        template<class Archive, class T>
        inline void load(Archive & ar, std::unique_ptr< T > &t, const unsigned int /*file_version*/){
            T *base_pointer;
            ar & BOOST_SERIALIZATION_NVP(base_pointer);
            t.reset(base_pointer);
        }
        template<class Archive, class T>
        inline void serialize(Archive & ar, std::unique_ptr< T > &t, const unsigned int file_version){
            boost::serialization::split_free(ar, t, file_version);
        }
    } // namespace serialization
} // namespace boost