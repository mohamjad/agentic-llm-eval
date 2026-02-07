# Security Policy

## Supported Versions

We currently support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT** open a public issue
2. Email security details to: [Your Email] (or create a private security advisory on GitHub)
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will respond within 48 hours and work with you to address the issue before public disclosure.

## Security Best Practices

When using this framework:

- **Never commit API keys or tokens** - Use environment variables or secure credential storage
- **Validate all inputs** - The framework includes validation utilities; use them
- **Review execution traces** - Check traces for sensitive data before logging
- **Keep dependencies updated** - Regularly update `requirements.txt` dependencies
- **Use HTTPS** - Always use secure connections for API calls

## Known Security Considerations

- The safety metric uses keyword detection which may have false positives/negatives
- Execution traces may contain sensitive information - review before sharing
- Configuration files may contain sensitive settings - keep them secure
