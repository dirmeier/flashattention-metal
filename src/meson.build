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
fa_generated = []
fa_deps = []

if metal_available
  metal_cpp_dep = dependency('metal_cpp', fallback : ['metal-cpp', 'metal_cpp_dep'])
  mlx_dep = dependency('MLX', method : 'cmake')

  flashattention_metallib = custom_target(
    'flashattention_metallib',
    input : 'flashattention.metal',
    output : 'flashattention.metallib',
    command : ['xcrun', 'metal', '@INPUT@', '-o', '@OUTPUT@'],
    build_by_default : true)

  fs = import('fs')
  shader_conf = configuration_data()
  shader_conf.set('FLASHATTENTION_METAL', fs.read('flashattention.metal'))
  shader_conf.set('FLASHATTENTION_MLX_METAL', fs.read('flashattention_mlx.metal'))

  fa_generated += configure_file(
    input : 'embedded_shaders.hpp.in',
    output : 'embedded_shaders.hpp',
    configuration : shader_conf)

  fa_sources += files([
    'metal_symbols.cpp',
    'metal_device.cpp',
    'flashattention.cpp',
    'flashattention_mlx.cpp'])
  fa_deps += [metal_cpp_dep, mlx_dep]
endif

fa_lib = static_library(
  'fa',
  fa_sources,
  fa_generated,
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
