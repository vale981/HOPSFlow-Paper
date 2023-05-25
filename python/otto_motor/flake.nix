{
  description = "Testing out the one bath case with time dependent coupling and two baths.";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    utils.url = "github:vale981/hiro-flake-utils";
    utils.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, utils, nixpkgs, ... }:
    (utils.lib.poetry2nixWrapper nixpkgs {
      name = "09_dynamic_two_bath_one_qubit";
      shellPackages = (pkgs:
        (with pkgs;
        [
          pyright
          python39Packages.jupyter
          sshfs
          arb
          (pkgs.texlive.combine {
            inherit (pkgs.texlive) scheme-medium
              type1cm unicode-math;
          })
        ]));

      python = pkgs: pkgs.python39Full;
      shellOverride = (pkgs: oldAttrs: {
        shellHook = ''
                        # export PYTHONPATH=/home/hiro/src/stocproc/:$PYTHONPATH
                        export PYTHONPATH=/home/hiro/src/two_qubit_model/:$PYTHONPATH
                        # export PYTHONPATH=/home/hiro/src/hops/:$PYTHONPATH
                        # export PYTHONPATH=/home/hiro/src/hopsflow/:$PYTHONPATH
                        export LD_LIBRARY_PATH="${(pkgs.lib.makeLibraryPath [pkgs.arb])}"
          #               '';
      });
      noPackage = true;
      poetryArgs = {
        projectDir = ./.;
      };
    });
}
