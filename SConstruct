#srodowisko dla debiana -*- mode: Python; -*-
import os, platform, subprocess

faif_name = 'faif'

ver_major = '0'
ver_minor = '35'
ver_compilation = '000'

os_platform=platform.system()

BOOST_INCLUDE_WINDOWS = 'C:\\boost_1_57_0'
BOOST_INCLUDE_WINDOWS_MINGW = '/C/msys64/opt/boost_1_57'
#BOOST_INCLUDE_LINUX = '/usr/local/include'
BOOST_INCLUDE_LINUX = '/usr/local/boost_1_57_0'
BOOST_LIB_WINDOWS_MINGW = '/C/msys64/opt/boost_1_57/stage/lib'
BOOST_LIB_WINDOWS = 'C:\\boost_1_57_0\\lib64-msvc-12.0'
BOOST_LIB_LINUX = '/usr/local/boost_1_57_0/stage/lib'
#BOOST_LIB_LINUX = '/usr/local/lib'

BOOST_THREAD_LINUX = 'boost_thread'
BOOST_THREAD_LINUX_D = 'boost_thread'
BOOST_DATE_TIME_LINUX = 'boost_date_time'
BOOST_DATE_TIME_LINUX_D = 'boost_date_time'
BOOST_CHRONO_LINUX = 'boost_chrono'
BOOST_CHRONO_LINUX_D = 'boost_chrono'
BOOST_SYSTEM_LINUX = 'boost_system'
BOOST_SYSTEM_LINUX_D = 'boost_system'
BOOST_SERIALIZATION_LINUX = 'boost_serialization'
BOOST_SERIALIZATION_LINUX_D = 'boost_serialization'
BOOST_UNIT_TEST_LINUX = 'boost_unit_test_framework'
BOOST_UNIT_TEST_LINUX_D = 'boost_unit_test_framework'

#odczytuje wersje kompilacji z wersji repozytorium
#ver_repository = subprocess.Popen('hg sum', shell=True, stdout=subprocess.PIPE).communicate()[0]
try:
   ver_compilation = re.search('(?<=parent: )\d+', ver_repository).group()
except BaseException:
   pass

try:   
	ver_compilation = subprocess.Popen('git rev-list --count HEAD', shell=True,stdout=subprocess.PIPE).communicate()[0]
	if (int(ver_compilation)<100):
		ver_compilation=`0`+ver_compilation.rstrip()
except BaseException:
   pass   
   
ver_install = '-1'

faif_ver = ver_major + '.' + ver_minor
faif_full_ver = faif_ver + '.' + ver_compilation + ver_install
#faif_full_ver='0.35.674' #todo git describe
#ver_compilation='100'
#dodaje dodatkowe argumenty do budowania

vars = Variables('custom.py')
vars.Add(BoolVariable('install', 'build debian package or setup.exe', 0))
vars.Add(EnumVariable('compiler','clang gcc and default','default',allowed_values={'clang','gcc','default'},map={},ignorecase=2))
vars.Add(EnumVariable('clibrary','c99 c11','c11',allowed_values={'c99','c11'},map={},ignorecase=2))

env = Environment(variables=vars)


#dodaje opis argumentow do pomocy
Help(vars.GenerateHelpText(env) )

Export('env')
Export('faif_name')
Export('ver_major ver_minor ver_compilation ver_install')


#settings for debug and release
if(os_platform== "Linux" or 'mingw' in os_platform.lower() ):
   if('mingw' in os_platform.lower()):

        env.Append( CPPPATH = [ Dir('.'), Dir(BOOST_INCLUDE_WINDOWS_MINGW) ] )
        env.Append( LIBPATH = BOOST_LIB_WINDOWS_MINGW )
   else:
        env.Append( CPPPATH = [ Dir('.'), Dir(BOOST_INCLUDE_LINUX) ] )
        env.Append( LIBPATH = BOOST_LIB_LINUX )


   #env.Append( CPPFLAGS = '-Wall -pedantic -pthread -fPIC -Wstrict-aliasing=2' )
   env.Append( CPPFLAGS = '-Wall -pedantic -pthread' )
   if (env['compiler']=='clang'):
        env.Replace(CXX='clang++') 
   elif (env['compiler']=='gcc'):
        env.Replace(CXX='gcc')

   if(env['clibrary']=='c11'):
        env.Append(CXXFLAGS='-std=c++11') 
   env.Append( LINKFLAGS = '-Wall -pthread' )
elif(os_platform== "Windows"):
   env.Append( CPPPATH = [ Dir('.'), Dir(BOOST_INCLUDE_WINDOWS) ] )
   env.Append( LIBPATH = BOOST_LIB_WINDOWS )
   env.Append( WINDOWS_INSERT_MANIFEST = True )
else:
   print "os_platform not supported"

envdebug = env.Clone()

env.debugFlag = False        #member used to distinguish between debug environment and non-debug one
envdebug.debugFlag = True

if(os_platform== "Linux" or 'mingw' in os_platform.lower()):
   env.Append( CPPFLAGS = ' -O3' )
   env.Append( LINKFLAGS = ' -O3' )
   env.Append( LIBS = [BOOST_THREAD_LINUX, BOOST_DATE_TIME_LINUX, BOOST_SERIALIZATION_LINUX, BOOST_CHRONO_LINUX, BOOST_SYSTEM_LINUX] )

   envdebug.Append( CPPFLAGS = ' -g ' ) #-ftest-coverage -fprofile-arcs
   envdebug.Append( LINKFLAGS = ' -g ' ) #-fprofile-arcs
   envdebug.Append( LIBS = [BOOST_THREAD_LINUX_D, BOOST_DATE_TIME_LINUX_D, BOOST_SERIALIZATION_LINUX_D, BOOST_CHRONO_LINUX_D, BOOST_SYSTEM_LINUX_D ] )

elif(os_platform== "Windows" ):
   env.Append( CPPFLAGS = ' /EHsc /MD /D "WIN32" /D "_CONSOLE" /W4 /Ox' )
   env.Append( LINKFLAGS = ' /SUBSYSTEM:CONSOLE ' )

   envdebug.Append( CPPFLAGS = ' /Od /EHsc /MDd /D "WIN32" /D "_CONSOLE" /D "_DEBUG" /W4 /ZI /TP' )
   envdebug.Append( LINKFLAGS = ' /SUBSYSTEM:CONSOLE /DEBUG ' )
else:
   print "System " + os_platform +" not supported "


def add_test_settings( e ):
   if(os_platform== "Linux" or 'mingw' in os_platform.lower()):
      if hasattr(e, 'debugFlag'):
         if e.debugFlag:
            e.Append( LIBS = BOOST_UNIT_TEST_LINUX_D )
         else:
            e.Append( LIBS = BOOST_UNIT_TEST_LINUX )
      else:
         e.Append( LIBS = BOOST_UNIT_TEST_LINUX )
   elif(os_platform== "Windows" ):
      pass
   else:
      print 'system not supported'
   return e

import files

#dokleja nazwe build dir na poczatku kazdego elementu z listy src_files
def prepare_src_files( build_dir, src_files):
   src_compilation = []
   for f in src_files:
      src_compilation.append(build_dir + f)
   return src_compilation

def build_library_version( target, source, env):
   file=open(str(target[0]),'w')
   file.write('// this file is generated by SCons script\n')
   file.write('#ifndef FAIF_VERSION_H\n')
   file.write('#define FAIF_VERSION_H\n')
   file.write('namespace faif {\n')
   file.write('  const int FAIF_VERSION_MAJOR = ' + ver_major + ';\n')
   file.write('  const int FAIF_VERSION_MINOR = ' + ver_minor + ';\n')
   file.write('  const int FAIF_VERSION_COMPILATION = ' + ver_compilation + ';\n')
   file.write('} //namespace faif\n')
   file.write('#endif //FAIF_VERSION_H\n')
   file.close()
   return

def build_library(env):
    #copy headers
    install_include_dir = './faif/'
    for mod in files.modules:
        dir = install_include_dir + mod.path
        for file in mod.head:
           env.Install(dir, 'src/' + file)
    return

def build_single_program( env, target, source):
   t = env.Program( target = target, source = source)
   env.SideEffect( File( str(target) + '.pdb'), t )
   env.SideEffect( File( str(target) + '.ilk'), t )
   env.SideEffect( File( str(target) + '.exp'), t )
   env.SideEffect( File( str(target) + '.lib'), t )
   return t

def build_tests( env, build_dir, post_name ):
   em = env.Clone()
   em = add_test_settings(em)
   em.Append( CPPFLAGS = ' -D BOOST_TEST_MAIN' )
   em.VariantDir( build_dir, 'tests/', duplicate = 0)
   #tests of given properties
   build_single_program( em, 'testPrimitives'+post_name, prepare_src_files(build_dir, ['PrimitivesTest.cpp'] ) )
   build_single_program( em, 'testUtils'+post_name, prepare_src_files(build_dir, [ 'UtilsTest.cpp'] ) )
   build_single_program( em, 'testDna'+post_name, prepare_src_files(build_dir, [ 'DnaTest.cpp'] ) )
   build_single_program( em, 'testHapl'+post_name, prepare_src_files(build_dir, [ 'HaplTest.cpp'] ) )
   build_single_program( em, 'testTimeseries'+post_name, prepare_src_files(build_dir, [ 'TimeseriesTest.cpp'] ) )
   build_single_program( em, 'testTSPred'+post_name, prepare_src_files(build_dir, [ 'TimeseriesPredictionTest.cpp'] ) )
   build_single_program( em, 'testLearning'+post_name, prepare_src_files(build_dir, [ 'LearningTest.cpp'] ) )
   build_single_program( em, 'testNBC'+post_name, prepare_src_files(build_dir, [ 'NbcTest.cpp'] ) )
   build_single_program( em, 'testDTC'+post_name, prepare_src_files(build_dir, [ 'DtcTest.cpp'] ) )
   build_single_program( em, 'testKNN'+post_name, prepare_src_files(build_dir, [ 'KnnTest.cpp'] ) )
   build_single_program( em, 'testSearch'+post_name, prepare_src_files(build_dir, [ 'SearchTest.cpp'] ) )
   build_single_program( em, 'testOptAlg'+post_name, prepare_src_files(build_dir, [ 'OptAlgTest.cpp'] ) )

   #all tests
   ea = env.Clone()
   ea = add_test_settings(ea)
   ea.VariantDir( build_dir + 'all/', 'tests/', duplicate = 0)
   build_single_program( ea, 'testAll'+post_name, prepare_src_files(build_dir+'all/',
                                                                    ['PrimitivesTest.cpp', 'UtilsTest.cpp',
                                                                     'DnaTest.cpp', 'HaplTest.cpp',
                                                                     'TimeseriesTest.cpp', 'TimeseriesPredictionTest.cpp',
                                                                     'LearningTest.cpp', 'NbcTest.cpp', 'DtcTest.cpp', 'KnnTest.cpp',
                                                                     'SearchTest.cpp', 'OptAlgTest.cpp',
                                                                     'TestAll.cpp' ] ) )
   return

def build_examples( env, build_dir ):
   em = env.Clone()
   em.VariantDir( build_dir, 'examples/', duplicate = 0)
   build_single_program(em, 'dna', source = prepare_src_files(build_dir, ['dna.cpp'] ) )
   build_single_program(em, 'ea', source = prepare_src_files(build_dir, ['ea.cpp'] ) )
   build_single_program(em, 'hillclimb', source = prepare_src_files(build_dir, ['hillclimb.cpp'] ) )
   build_single_program(em, 'nbc', source = prepare_src_files(build_dir, ['nbc.cpp'] ) )
   build_single_program(em, 'dtc', source = prepare_src_files(build_dir, ['dtc.cpp'] ) )
   build_single_program(em, 'knn', source = prepare_src_files(build_dir, ['knn.cpp'] ) )
   build_single_program(em, 'nbcdb', source = prepare_src_files(build_dir, ['nbcdb.cpp'] ) )
   build_single_program(em, 'dtcdb', source = prepare_src_files(build_dir, ['dtcdb.cpp'] ) )
   build_single_program(em, 'random', source = prepare_src_files(build_dir, ['random.cpp'] ) )
   build_single_program(em, 'search', source = prepare_src_files(build_dir, ['search.cpp'] ) )


if env['install'] == 1:
   SConscript('install/SConscript')
else:
   file_ver_name = 'src/Version.hpp'
   env.Command(file_ver_name, [], build_library_version )
   env.SideEffect( 'vc100.idb', 'src/Version.hpp' )
   env.SideEffect( 'vc100.pdb', 'src/Version.hpp' )
   env.SideEffect( 'install/faif-src-'+faif_ver+'.tar', 'src/Version.hpp' )
   env.SideEffect( 'install/faif-src-'+faif_ver+'.tar.bz2', 'src/Version.hpp' )
   env.SideEffect( 'install/faif-doc-'+faif_ver+'.tar', 'src/Version.hpp' )
   env.SideEffect( 'install/faif-doc-'+faif_ver+'.tar.bz2', 'src/Version.hpp' )
   env.SideEffect( 'install/' + faif_name + '_'+ faif_full_ver + '_i386.deb', 'src/Version.hpp' )
   env.SideEffect( 'install/' + faif_name + '_'+ faif_full_ver + '_amd64.deb', 'src/Version.hpp' )
   env.SideEffect( 'install/Faif-' + faif_full_ver + '-setup.exe', 'src/Version.hpp' )
   build_tests( env, 'build/test/', '')
   build_tests( envdebug, 'build/test_d/', '-d')
   build_library( env )
   build_examples( env, 'build/examples/' )