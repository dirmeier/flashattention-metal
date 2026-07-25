metal_available = false
if not get_option('metal').disabled()
  metal_check = run_command('xcrun', '--find', 'metal', check : false)
  metal_available = (metal_check.returncode() == 0)
endif

if get_option('metal').enabled() and not metal_available
  error('"xcrun --find metal" failed. Install the full Xcode toolchain.')
endif

fa_sources = files([
  'matrix.cpp',
  'attention_naive.cpp'])
fa_deps = []

if metal_available
  metal_cpp_dep = dependency('metal_cpp', fallback : ['metal-cpp', 'metal_cpp_dep'])

  flashattention_metallib = custom_target(
    'flashattention_metallib',
    input : 'flashattention.metal',
    output : 'flashattention.metallib',
    command : ['xcrun', 'metal', '@INPUT@', '-o', '@OUTPUT@'],
    build_by_default : true)

  fa_sources += files([
    'metal_symbols.cpp',
    'metal_device.cpp',
    'flashattention.cpp'])
  fa_deps += metal_cpp_dep
endif

fa_lib = static_library(
  'fa',
  fa_sources,
  include_directories : inc,
  cpp_args : warning_flags,
  dependencies : fa_deps)

if metal_available
  executable(
    'flash-attention',
    'main.cpp',
    include_directories : inc,
    cpp_args : warning_flags,
    link_with : fa_lib,
    dependencies : fa_deps)
endif
