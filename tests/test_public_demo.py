import os
from pathlib import Path

os.environ["FILMTRACE_DATASET"] = "movielens-32m-21c"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from streamlit.testing.v1 import AppTest

from cloud_data import load_public_data, poster_url
from src.personal_feedback import Profile
from src.recommendation_runtime import ALGORITHMS, RecommendationRuntime

ROOT = Path(__file__).resolve().parents[1]


def test_public_dataset_and_algorithms(tmp_path):
    ratings, movies, discovery, manifest = load_public_data()
    assert (ratings.user_id.nunique(), len(movies), len(ratings)) == (1200, 1054, 96848)
    assert movies.release_year.min() >= 2001
    runtime = RecommendationRuntime(ratings, movies, discovery, cache_dir=tmp_path)
    keys = runtime.catalog.sort_values("Ratings", ascending=False).Key.head(3).tolist()
    profile = Profile(dict(zip(keys, [5, 4, 1])))
    for algorithm in ALGORITHMS:
        rows = runtime.recommend(profile, algorithm, 5)
        assert len(rows) == 5 and not set(rows.Key) & set(keys)
        profile.blocked.add(rows.Key.iloc[0])
        again = runtime.recommend(profile, algorithm, 5)
        assert not set(again.Key) & profile.blocked
    assert manifest["new_release_records"] > 0


def test_only_approved_poster_urls():
    assert poster_url("https://image.tmdb.org/t/p/w500/example.jpg").endswith("/w342/example.jpg")
    for value in [None, float("nan"), "file:///etc/passwd", "https://example.com/image.jpg"]:
        assert poster_url(value) is None


def test_visitor_rate_search_reset_and_session_isolation():
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60).run()
    assert not app.exception
    app.radio(key="visitor_view").set_value("Explore & rate").run()
    save = next(button for button in app.button if button.label == "Save rating")
    save.click().run()
    assert len(app.session_state["visitor_profile"].ratings) == 1
    other = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60).run()
    assert not other.session_state["visitor_profile"].ratings
    for query in ["[", ".*", "+12"]:
        app.text_input(key="visitor_query").set_value(query).run()
        assert not app.exception
    app.radio(key="visitor_view").set_value("New movies for you").run()
    assert not app.exception
    next(button for button in app.button if button.label == "Reset visitor").click().run()
    assert not app.exception and not app.session_state["visitor_profile"].ratings
    assert not (ROOT / "data/private").exists()
