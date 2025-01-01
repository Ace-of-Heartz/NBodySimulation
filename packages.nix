# The items in the curly brackets are function parameters as this is a Nix
# function that accepts dependency inputs and returns a new package
# description
{ lib
, stdenv
, cmake
, ninja
, ocl-icd
, opencl-headers
, opencl-clhpp
, SDL2
, SDL2_image
, glm
, glew
, imgui
, libGLU
}:

# stdenv.mkDerivation now accepts a list of named parameters that describe
# the package itself.

stdenv.mkDerivation {
  name = "cpp-nix";

  # good source filtering is important for caching of builds.
  # It's easier when subprojects have their own distinct subfolders.
  src = lib.sourceByRegex ./. [
    "^src.*"
    "^test.*"
    "CMakeLists.txt"
  ];

  # We now list the dependencies similar to the devShell before.
  # Distinguishing between `nativeBuildInputs` (runnable on the host
  # at compile time) and normal `buildInputs` (runnable on target
  # platform at run time) is an important preparation for cross-compilation.
  nativeBuildInputs = [ cmake ninja ];
  buildInputs = [
    ocl-icd
    opencl-headers
    opencl-clhpp
    glm
    glew
    libGLU
    imgui
    SDL2
    SDL2_image
    ];

  installPhase = ''
      mkdir -p $out/bin
      cp NBody $out/bin
    '';
  # Instruct the build process to run tests.
  # The generic builder script of `mkDerivation` handles all the default
  # command lines of several build systems, so it knows how to run our tests.
  doCheck = true;
}