# Argus — Project Page

This branch (`gh-pages`) hosts the static project page for
**Argus: Metric Panoramic 3D Reconstruction for Indoor Scenes**.

Live site: https://realsee-developer.github.io/Argus/

## Structure
```
index.html              # main page
static/css/index.css    # styles
static/js/index.js      # BibTeX copy button
static/images/*.png     # figures exported from the paper
.nojekyll               # disable Jekyll processing
```

## Local preview
```bash
python3 -m http.server 8000
# open http://localhost:8000
```

## Notes
- All asset paths are **relative** (`./static/...`) so the site works under the
  `/Argus/` project sub-path as well as on a custom domain.
- To use a custom domain, add a `CNAME` file containing the domain and configure DNS.
- The website template is adapted from [Nerfies](https://github.com/nerfies/nerfies.github.io)
  (CC BY-SA 4.0).

This branch is independent of `main`; the source code and paper live on `main` and are unaffected.
