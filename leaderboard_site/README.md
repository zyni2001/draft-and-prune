# Reasoner Arena

Static leaderboard site for the AR-LSAT adaptive-agent benchmark.

## Run locally

From the repository root:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/leaderboard_site/
```

## Deploy with GitHub Pages

This repository now includes:

```text
.github/workflows/deploy-leaderboard-pages.yml
```

It publishes the contents of:

```text
leaderboard_site/
```

To enable it:

1. Push the workflow to GitHub.
2. In the repository settings, open `Pages`.
3. Set the source to `GitHub Actions`.
4. Push to `main` or `master`, or manually run the workflow from the Actions tab.

After deployment, the site will be served from your repository's GitHub Pages URL.

## Data source

The page reads from:

```text
leaderboard_site/data/leaderboard.json
```

Update that file to refresh the displayed snapshot without changing the UI.
