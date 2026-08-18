\newpage

# Colophon

The code accompanying this book lives at **github.com/aelena/c-tiktoken**, MIT licensed. It builds with any C23-capable compiler and depends only on PCRE2. The integration suite compares its output against Python's `tiktoken` package, token for token, across a corpus that includes the cases most implementations get wrong: embedded nulls, multi-byte UTF-8 at chunk boundaries, special tokens, and the empty string.

If you find a divergence from the reference implementation, that is a bug and I would like to know about it. Open an issue.

## About the author

Antonio Elena is CTO at BEAI Energy. Over two decades across enterprise, consulting and startups he has worked as an architect, engineering leader and CTO, and was previously Global Head of Architecture & Technology at SGS, reporting to the Group CIO.

He writes about architecture, cloud economics and AI strategy at **sig-intent.com**, and takes on selective advisory work through the same site: fractional CTO, architecture review and AI strategy.

## Also available

**XML-Structured Prompting: A Software Engineering Discipline for Reliable LLM Systems**. Prompting treated as an engineering practice rather than a style preference. Vocabulary, constraints and output contracts as design interfaces; versioning, testing and governance for prompts that live in production.

Available at **aelena74.gumroad.com/l/xsp**.

---

*Set in Georgia and Consolas. Built with WeasyPrint from the Markdown sources in the repository, which means this PDF and the tutorial on GitHub cannot drift apart.*
