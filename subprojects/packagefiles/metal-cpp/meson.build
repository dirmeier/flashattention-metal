project('metal-cpp', 'cpp', version : '26.4', default_options : ['cpp_std=c++20'])

metal_cpp_dep = declare_dependency(
  include_directories : include_directories('.'),
  dependencies : [
    dependency('appleframeworks', modules : ['Metal', 'Foundation', 'QuartzCore']),
  ])
