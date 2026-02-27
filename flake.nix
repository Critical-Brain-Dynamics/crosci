{
  description = "Python development environment with uv";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # Python version from pyproject.toml requires-python
        # This will be managed by uv, not nix directly
        pythonVersion = "3.14"; # ADJUST based on pyproject.toml
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # uv for Python package and environment management
            uv
            gcc
            llvm
            libgccjit
            cmake

            # Optional: useful development tools
            git
          ];

          shellHook = ''
            # Set up uv to use the correct Python version
            export UV_PYTHON_PREFERENCE=only-managed
            export UV_PYTHON=${pythonVersion}

            # Create virtual environment if it doesn't exist
            if [ ! -d ".venv" ]; then
              echo "Creating virtual environment with Python ${pythonVersion}..."
              uv venv --python ${pythonVersion}
            fi

            # Activate the virtual environment
            source .venv/bin/activate

            # Sync dependencies if pyproject.toml exists
            if [ -f "pyproject.toml" ]; then
              echo "Syncing dependencies..."
              uv sync --all-extras 2>/dev/null || uv pip install -e ".[dev]" 2>/dev/null || true
            fi

            # Install pre-commit if exists
            if [ -f ".pre-commit-config.yaml" ]; then
              echo "Installing pre-commit hooks"
              uv run pre-commit install
            fi

            echo "Python environment ready: $(python --version)"
          '';
        };
      }
    );
}
