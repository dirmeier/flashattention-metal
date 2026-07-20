project(
  'metal',
  'cpp',
  version : '0.1.0',
  default_options : [
    'buildtype=release',
    'warning_level=3',
    'cpp_std=c++23',
    'b_ndebug=if-release'],
  meson_version : '>= 1.0.0')

cpp = meson.get_compiler('cpp')

warning_flags = []
foreach cflag: [
  '-Wdouble-promotion',
  '-Wnull-dereference',
  '-Wshadow',
  '-Wformat=1']
    if cpp.has_argument(cflag)
      warning_flags += cflag
    endif
endforeach

inc = include_directories('include')


subdir('src')
