# Security Policy

We currently support versions 0.2.x and 0.1.x with security updates. Versions below 0.1 are not supported.

If you discover a security vulnerability, report it responsibly. Do not open a public issue. Email security details or create a private security advisory on GitHub. Include description of the vulnerability, steps to reproduce, potential impact, and suggested fix if any. We will respond within 48 hours and work with you to address the issue before public disclosure.

Security best practices when using this framework include never committing API keys or tokens, using environment variables or secure credential storage instead, validating all inputs using the framework's validation utilities, reviewing execution traces for sensitive data before logging, keeping dependencies updated by regularly updating requirements.txt dependencies, and using HTTPS for all API calls.

Known security considerations include the safety metric using keyword detection which may have false positives or false negatives, execution traces potentially containing sensitive information that should be reviewed before sharing, and configuration files potentially containing sensitive settings that should be kept secure.
