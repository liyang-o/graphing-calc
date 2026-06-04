# Web Graphing Calculator

A browser-based interactive graphing calculator built with TypeScript, Vite, and Canvas.

## Features

- Plot multiple expressions in `x`
- Mouse wheel zoom at the cursor
- Drag to pan the coordinate plane
- Double-click to reset the view
- Add, disable, and delete expressions
- Export the current graph as PNG
- Supports common math functions: `sin`, `cos`, `tan`, `sqrt`, `abs`, `ln`, `log`, `exp`, `min`, `max`, `pow`
- Supports constants `pi`, `π`, `e`, `tau`
- Supports implicit multiplication, such as `2x`, `2sin(x)`, `(x+1)(x-1)`

## Run locally

```bash
npm install
npm run dev
```

Then open the URL printed by Vite.

## Test and build

```bash
npm run test
npm run build
```

## Deploy to GitHub Pages

The project includes a GitHub Actions workflow at `.github/workflows/deploy-pages.yml`.

1. In GitHub, open `Settings -> Pages`.
2. Set `Build and deployment -> Source` to `GitHub Actions`.
3. Push to `main` or run the workflow manually.

The expected Pages URL is:

```text
https://liyang-o.github.io/graphing-calc/
```
