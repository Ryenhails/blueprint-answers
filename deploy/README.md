# Hosting the live demo

The published page (`docs/index.html`, GitHub Pages) only holds stored answers. To let other people upload their own
drawings, run the server in hosted mode. It serves the same stored examples and answers uploads live.

```bash
docker build -t blueprint-answers .
docker run -p 7860:7860 -e GEMINI_API_KEY=... blueprint-answers
```

- Open by default, with limits per visitor address: `STOA_ASKS_PER_HOUR` (default 40) and `STOA_UPLOADS_PER_HOUR` (default 10).
- `STOA_ACCESS_CODE` (optional): when set, uploads and live questions also need this code.
- `STOA_BUDGET_USD` (default 25): the server stops calling models after this much API spend since it started.
- Uploaded files stay in a temporary folder of the container and are gone when it restarts.

## Hugging Face Space

1. Create a Space with the Docker SDK.
2. In the Space settings add the secret `GEMINI_API_KEY`.
3. Push this repository to the Space, with `deploy/space_README.md` as its `README.md`:

```bash
git clone https://github.com/Ryenhails/blueprint-answers && cd blueprint-answers
cp deploy/space_README.md README.md && git commit -am "Space front matter"
git remote add space https://huggingface.co/spaces/<user>/blueprint-answers
git push space main --force
```
