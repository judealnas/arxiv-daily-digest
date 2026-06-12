#!/usr/bin/env python3
"""
Send the arXiv digest HTML file as a multipart/alternative email (plain text + HTML).

Designed for no_agent cron jobs. Reads credentials from Hermes .env file.
Usage: arxiv-email-sender.py [--to EMAIL] [--html PATH] [--subject SUBJECT]

Defaults:
  --to      jude.alnas@duke.edu
  --html    ~/.hermes/cron/shared/arxiv-digest.html
  --subject "arXiv quant-ph daily digest — YYYY-MM-DD"
"""

import argparse
import os
import re
import smtplib
import ssl
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate
from pathlib import Path
from datetime import datetime, timezone, timedelta


def load_env(hermes_home: str) -> dict:
    """Load EMAIL_* vars from ~/.hermes/.env"""
    env_path = Path(hermes_home) / ".env"
    env = {}
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    env[key] = val
    return env


def strip_html(html: str) -> str:
    """Convert HTML to readable plain text."""
    # Replace <br> and <p> with newlines
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    text = re.sub(r"<p[^>]*>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<li[^>]*>", "\n- ", text, flags=re.IGNORECASE)
    text = re.sub(r"</li>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<h\d[^>]*>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</h\d>", "", text, flags=re.IGNORECASE)
    # Replace <a href="URL">TEXT</a> with TEXT (URL)
    text = re.sub(r'<a\s+href="([^"]+)"[^>]*>([^<]+)</a>', r"\2 (\1)", text, flags=re.IGNORECASE)
    # Remove remaining tags
    text = re.sub(r"<[^>]+>", "", text)
    # Decode entities
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&quot;", '"', text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_authors_from_html(html: str) -> str:
    """Remove author names from arXiv paper entries in HTML.

    Matched papers:   <a href="...">Title</a> — Summary...
    Non-matched:      <a href="...">Title</a> — Author1, A2<br/>
    """
    # Remove author lists after </a> — and before <br/>
    html = re.sub(
        r'(</a>\s*—\s*)[^<]*?(<br\s*/?>)',
        r'\1\2',
        html,
    )
    return html


def strip_arxiv_version(url: str) -> str:
    """Remove version suffix from arXiv URLs for cleaner display."""
    return re.sub(r"v\d+$", "", url)


def send_email(
    to_addr: str,
    subject: str,
    html_body: str,
    plain_body: str,
    smtp_host: str,
    smtp_port: int,
    username: str,
    password: str,
) -> None:
    """Send multipart/alternative email via SMTP."""
    msg = MIMEMultipart("alternative")
    msg["From"] = username
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = f"<arxiv-digest-{datetime.now().strftime('%Y%m%d%H%M%S')}@{username.split('@')[1]}>"
    msg["X-Mailer"] = "Hermes Agent arXiv Digest"
    msg["Precedence"] = "bulk"  # Let receivers know this is automated
    msg["Auto-Submitted"] = "auto-generated"

    # Attach plain text version first (fallback)
    msg.attach(MIMEText(plain_body, "plain", "utf-8"))
    # Attach HTML version (preferred)
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    # Send via SMTP with STARTTLS
    smtp = smtplib.SMTP(smtp_host, smtp_port, timeout=30)
    try:
        smtp.ehlo()
        smtp.starttls(context=ssl.create_default_context())
        smtp.ehlo()
        smtp.login(username, password)
        smtp.send_message(msg)
        print(f"Email sent to {to_addr}: {subject}")
    finally:
        try:
            smtp.quit()
        except Exception:
            smtp.close()


def main():
    hermes_home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
    env = load_env(hermes_home)

    # Determine yesterday's date (the digest subject)
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

    parser = argparse.ArgumentParser(description="Send arXiv digest email")
    parser.add_argument("--to", default="jude.alnas@duke.edu")
    parser.add_argument("--html", default=str(Path(hermes_home) / "cron" / "shared" / "arxiv-digest.html"))
    parser.add_argument("--subject", default=f"arXiv quant-ph daily digest — {yesterday}")
    parser.add_argument("--dry-run", action="store_true", help="Print email content instead of sending")
    args = parser.parse_args()

    # Read HTML digest
    html_path = Path(args.html)
    if not html_path.exists():
        print(f"ERROR: HTML digest not found: {html_path}", file=sys.stderr)
        sys.exit(1)

    raw_html = html_path.read_text(encoding="utf-8").strip()
    if not raw_html:
        print("ERROR: HTML digest is empty", file=sys.stderr)
        sys.exit(1)

    # Generate plain text version
    plain = strip_html(raw_html)

    # Get SMTP credentials
    smtp_host = env.get("EMAIL_SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(env.get("EMAIL_SMTP_PORT", "587"))
    username = env.get("EMAIL_ADDRESS", "")
    password = env.get("EMAIL_PASSWORD", "")

    if not username or not password:
        print("ERROR: EMAIL_ADDRESS or EMAIL_PASSWORD not found in .env", file=sys.stderr)
        sys.exit(1)

    # Strip existing html/body tags from raw content to avoid double-wrapping
    body_only = raw_html
    body_only = re.sub(r"^<!DOCTYPE[^>]*>\s*", "", body_only, flags=re.IGNORECASE)
    body_only = re.sub(r"^<html[^>]*>\s*", "", body_only, flags=re.IGNORECASE)
    body_only = re.sub(r"^<body[^>]*>\s*", "", body_only, flags=re.IGNORECASE)
    body_only = re.sub(r"\s*</body>\s*$", "", body_only, flags=re.IGNORECASE)
    body_only = re.sub(r"\s*</html>\s*$", "", body_only, flags=re.IGNORECASE)

    # Remove author lists (belt-and-suspenders: catches any the LLM left in)
    body_only = strip_authors_from_html(body_only)

    # Build final HTML with head, meta, and styles for email client compatibility
    html_email = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<!--[if mso]>
<style type="text/css">
  body {{ font-family: Arial, Helvetica, sans-serif; }}
  table {{ border-collapse: collapse; }}
  td {{ font-family: Arial, Helvetica, sans-serif; }}
</style>
<![endif]-->
<style type="text/css">
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; font-size: 14px; line-height: 1.5; color: #1f2937; padding: 20px; }}
  a {{ color: #2563eb; text-decoration: underline; word-break: break-word; }}
  h1 {{ font-size: 22px; color: #111827; border-bottom: 2px solid #e5e7eb; padding-bottom: 8px; }}
  h2 {{ font-size: 18px; color: #374151; margin-top: 24px; }}
  .paper {{ margin-bottom: 16px; padding: 12px; border-left: 3px solid #e5e7eb; }}
  .paper.matched {{ border-left-color: #f59e0b; background: #fffbeb; }}
  .paper-id {{ color: #6b7280; font-size: 12px; }}
</style>
</head>
<body>
{body_only}
</body>
</html>"""

    if args.dry_run:
        print(f"=== DRY RUN: Would send to {args.to} ===")
        print(f"Subject: {args.subject}")
        print(f"HTML length: {len(html_email)} chars")
        print(f"Plain length: {len(plain)} chars")
        print()
        print("=== PLAIN TEXT PREVIEW (first 2000 chars) ===")
        print(plain[:2000])
        print()
        print("=== HTML PREVIEW (first 2000 chars) ===")
        print(html_email[:2000])
        return 0

    send_email(
        to_addr=args.to,
        subject=args.subject,
        html_body=html_email,
        plain_body=plain,
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        username=username,
        password=password,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
