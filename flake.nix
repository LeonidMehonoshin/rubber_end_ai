{
  outputs = { self, nixpkgs }: {
    devShells = builtins.mapAttrs (system: pkgs: {
      default = pkgs.mkShell {
        buildInputs = with pkgs; [
          (python3.withPackages (ps: with ps; [
            torch
            pandas
            scikit-learn
            cryptography
            pyside6
          ]))
        ];
      };
    }) nixpkgs.legacyPackages;
  };

  description = "Python development environment";
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
}
