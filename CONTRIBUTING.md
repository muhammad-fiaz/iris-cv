# Contributing to Iris

Thank you for your interest in contributing to Iris! As a pure Rust computer vision ecosystem powered by Burn, we welcome contributions of all kinds, including bug fixes, feature requests, new operators, and documentation improvements.

## Code of Conduct

We expect all contributors to adhere to standard respectful collaboration guidelines.

## Development Workflow

1. **Fork and Clone**: Fork the repository on GitHub and clone your fork locally.
2. **Setup Rust**: Make sure you are using the latest stable Rust compiler (Rust 2024 edition is required).
3. **Build the Project**: Run `cargo build` to build the library.
4. **Run Tests**: Before making changes, verify that the existing tests pass with `cargo test`.
5. **Implement Changes**: Add your code changes, and make sure to include unit tests and/or examples if introducing new features.
6. **Lint and Format**:
   - Run `cargo fmt` to auto-format your code.
   - Run `cargo clippy --all-targets` to check for lints and warnings. Your code should compile with no warnings.
7. **Commit and Push**: Write clear, descriptive commit messages, and push to your fork.
8. **Submit a Pull Request**: Open a pull request against the `main` branch of the main repository.

## Coding Style

- Follow standard Rust formatting conventions.
- Document public APIs using Rustdoc comments (`///`).
- Keep code clean, modular, and performant. Avoid unnecessary allocations inside tight loops.
