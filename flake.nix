{
  description = "OpenCL Development for GPGPU";

  inputs = {
    nixpkgs = {
        url = "github:NixOS/nixpkgs/nixos-unstable";
    };
  };

  outputs = inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      # This is the list of architectures that work with this project
      systems = [
        "x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin"
      ];
      perSystem = { config, self', inputs', pkgs, system, ... }: {
        # The `callPackage` automatically fills the parameters of the function
        # in package.nix with what's inside the `pkgs` attribute.
        packages.default = pkgs.callPackage ./packages.nix { };

        # The `config` variable contains our own outputs, so we can reference
        # neighbor attributes like the package we just defined one line earlier.
        devShells.default = config.packages.default;
      };
    };
}