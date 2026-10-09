# FilmTrace: Public Movie Discovery Demo

Public, account-free edition of FilmTrace for a university Open Day. It reuses the
corrected UserCF, ItemCF, SVD and content-ranking implementation from the local
project. The administrator interface and all private account data are excluded.

Run `streamlit run streamlit_app.py`. Cloud entry point: `streamlit_app.py`.
Use Python 3.11-3.14. No API key is needed: the catalogue includes cached
public movie metadata and validated TMDB image URLs. Posters require an internet
connection; recommendation calculations use the bundled data locally.

Visitor ratings are session-only and not written to an account database. Reset
visitor clears ratings, exclusions and preferences. Each visitor has a separate
Streamlit session; the immutable dataset and bounded model caches are shared.
Only fixed model parameters are exposed, and only the active view is computed.

## Dataset and Attribution

- GroupLens MovieLens 32M, bounded subset: 1,200 users, 1,054 movies, 96,848 ratings.
- Training catalogue contains films released from 2001 to 2023. The separate
  cached TMDB discovery catalogue contains newer releases; aggregate community
  scores are never presented as personal predictions.
- The public data bundle is gzip-compressed JSON encoded as base64 for transport.
  Its SHA-256 and counts are recorded in `data/manifest.json`. It contains only
  public dataset CSVs and whitelisted movie metadata, not application databases.
- [Original dataset and usage conditions](https://files.grouplens.org/datasets/movielens/ml-32m-README.html).
  Redistribution of this transformed subset remains subject to those same terms.
  Research use only; commercial or revenue-bearing use needs GroupLens permission.
  No endorsement by the University of Minnesota or GroupLens is implied.
- Cite: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
  History and Context. ACM TiiS 5(4), Article 19. https://doi.org/10.1145/2827872
- This product uses the TMDB API but is not endorsed or certified by TMDB.
  Film artwork belongs to its respective owners.

This repository is the public demonstration build, not the full administrator
application. Results demonstrate recommendation techniques, not guaranteed taste
predictions or independently measured post-fix accuracy claims.
