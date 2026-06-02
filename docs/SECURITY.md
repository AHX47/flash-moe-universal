# Security Policy

## Supported Versions

We currently support the `main` branch with the latest security updates.

| Version | Supported          |
| ------- | ------------------ |
| main    | ✅                 |
| < 1.0   | ❌ (development)   |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to [ahx47@example.com] (replace with your actual email). You should receive a response within 48 hours. If the issue is confirmed, we will release a patch as soon as possible.

## Security Considerations

- The engine loads model weights from disk – ensure you trust the model source.
- The HTTP server is intended for local use only; do not expose it to the internet without additional security layers (authentication, TLS, etc.).
- Memory usage is static after initialization – no dynamic allocations during generation, reducing risk of memory corruption.

## Disclosure Policy

When we receive a security report, we will:
1. Confirm the issue and determine affected versions.
2. Develop a fix and test it.
3. Release a patch and publicly disclose the issue after users have had time to update.

Thank you for helping keep Flash-MoE Universal secure.
