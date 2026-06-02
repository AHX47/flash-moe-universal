

# Contributing to Flash-MoE Universal

First off, thank you for considering contributing! It's people like you that make open source great.

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Check if the issue already exists in [GitHub Issues](https://github.com/AHX47/flash-moe-universal/issues). If not, open a new issue with:
- A clear title and description
- OS, architecture, compiler version
- Exact command that failed
- Relevant logs or screenshots

### Suggesting Enhancements

Open a feature request issue. Tell us:
- What problem it solves
- How you envision it working
- Any alternatives you've considered

### Pull Requests

1. Fork the repository and create your branch from `main`.
2. If you add code, add tests that cover it.
3. Ensure the test suite passes.
4. Update the documentation.
5. Issue a Pull Request with a clear description of changes.

## Development Setup

```bash
git clone https://github.com/AHX47/flash-moe-universal
cd flash-moe-universal
./build.sh          # Linux/macOS
# or build.bat on Windows
Code Style
C11 standard, no compiler-specific extensions where possible.

Indentation: 4 spaces, no tabs.

Function names: snake_case.

Use the platform abstraction (platform.h) instead of direct OS calls.

Adding a New Backend (GPU)
Create a file in src/backends/ (e.g., vulkan_backend.c).

Implement required functions: dequant_matvec, swiglu, rms_norm, etc.

Update CMakeLists.txt with option(USE_VULKAN ...) and conditional compilation.

Extend InferOptions and infer_create to select the backend.

Testing
Run the test suite with a small model (e.g., SmolLM2-360M):

bash
python scripts/download_model.py --model smollm2-360m
python scripts/test_model.py --model ./models/smollm2-360m
License
By contributing, you agree that your contributions will be licensed under the MIT License.

text

---

## 2. `CODE_OF_CONDUCT.md`

```markdown
# Contributor Covenant Code of Conduct

## Our Pledge

We as members, contributors, and leaders pledge to make participation in our community a harassment-free experience for everyone, regardless of age, body size, visible or invisible disability, ethnicity, sex characteristics, gender identity and expression, level of experience, education, socio-economic status, nationality, personal appearance, race, religion, or sexual identity and orientation.

We pledge to act and interact in ways that contribute to an open, welcoming, diverse, inclusive, and healthy community.

## Our Standards

Examples of behavior that contributes to a positive environment:
- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community

Examples of unacceptable behavior:
- The use of sexualized language or imagery
- Trolling, insulting/derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others' private information without explicit permission

## Enforcement Responsibilities

Project maintainers are responsible for clarifying and enforcing our standards of acceptable behavior and will take appropriate and fair corrective action in response to any behavior that they deem inappropriate, threatening, offensive, or harmful.

## Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported to the project team at [abdo47hak47@gmail.com]. All complaints will be reviewed and investigated promptly and fairly.

## Attribution

This Code of Conduct is adapted from the [Contributor Covenant][homepage], version 2.0, available at https://www.contributor-covenant.org/version/2/0/code_of_conduct.html.

[homepage]: https://www.contributor-covenant.org
