{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python313
    (python313.withPackages (ps: with ps; [
      torch
      pandas
      pyyaml
      scikit-learn
      cryptography
      pyside6
    ]))
    qt6.qtwayland
    qt6.qtbase
  ];
  shellHook = ''
    export QT_QPA_PLATFORM="wayland;xcb"
    export QT_IM_MODULE=compose
  '';
}
