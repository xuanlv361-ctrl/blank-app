"""Public, session-only FilmTrace demo. No account or admin entry points."""

import os

os.environ["FILMTRACE_DATASET"] = "movielens-32m-21c"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd
import streamlit as st

from cloud_data import load_public_data, poster_url
from src.catalog_search import catalog_genres, filter_catalog
from src.personal_feedback import Profile
from src.recommendation_runtime import ALGORITHMS, RecommendationRuntime

st.set_page_config(page_title="FilmTrace | Movie discovery", page_icon=":clapper:", layout="wide")


@st.cache_resource(show_spinner="Preparing the movie catalogue...")
def resources():
    ratings, movies, discovery, manifest = load_public_data()
    runtime = RecommendationRuntime(ratings, movies, discovery)
    posters = {f"catalog:{int(r.movie_id)}": r.poster_url for r in movies.itertuples()}
    posters.update({f"tmdb:{int(r['tmdb_id'])}": r.get("poster_url", "") for r in discovery})
    return runtime, posters, manifest


def reset_visitor():
    for key in list(st.session_state):
        if str(key).startswith("visitor_"):
            del st.session_state[key]


runtime, posters, manifest = resources()
profile = st.session_state.setdefault("visitor_profile", Profile())

st.markdown("""<style>
.stApp { background: #f9f9f9; color: #0f0f0f; }
[data-testid="stImage"] img { height: 245px; object-fit: contain; }
h1,h2,h3,p { letter-spacing: 0 !important; }
.stButton > button { min-height: 40px; border-radius: 8px; }
</style>""", unsafe_allow_html=True)
st.title("FilmTrace")
st.caption("Movie discovery | Open Day edition")

with st.sidebar:
    st.header("Your session")
    algorithm = st.selectbox("Algorithm", ALGORITHMS, key="visitor_algorithm")
    top_n = st.selectbox("Top N", [5, 10], key="visitor_top_n")
    st.metric("Your ratings", len(profile.ratings))
    st.button("Reset visitor", on_click=reset_visitor, use_container_width=True)
    if st.button("Preload models", use_container_width=True):
        with st.spinner("Preparing the three models..."):
            runtime.prewarm()
        st.success("Models ready")
    with st.expander("Data and privacy"):
        st.write("No account required. Your ratings and preferences are temporary and are not written to an account database.")
        st.write(f"{manifest['users']:,} dataset users / {manifest['movies']:,} movies / {manifest['ratings']:,} ratings.")
        st.write("MovieLens films: 2001-2023. New releases use a separate TMDB metadata catalogue.")
        st.link_button("MovieLens source and licence", "https://files.grouplens.org/datasets/movielens/ml-32m-README.html")


def cards(rows, scope, rate=False):
    if rows.empty:
        st.info("No films match this selection.")
        return
    for start in range(0, len(rows), 3):
        for col, row in zip(st.columns(3), rows.iloc[start:start + 3].to_dict("records")):
            with col:
                url = poster_url(posters.get(row["Key"]))
                if url:
                    st.image(url, use_container_width=True)
                else:
                    st.caption("Poster unavailable")
                st.subheader(row["Title"])
                st.caption(row["Genres"])
                if "Predicted Rating" in row and pd.notna(row["Predicted Rating"]):
                    st.write(f"Predicted rating: {row['Predicted Rating']:.2f}/5")
                elif "Content Match" in row and str(row.get("Score Type", "")).startswith("Content"):
                    st.write(f"Content match: {row['Content Match']:.3f}")
                elif pd.notna(row.get("Average Rating")):
                    st.write(f"Community rating: {row['Average Rating']:.2f}/5")
                if "Support Strength" in row:
                    st.caption(f"Support: {row['Support Strength']} ({row['Support Count']}); not a probability")
                with st.expander("Details / Why this movie?"):
                    st.write(row.get("Reason", "Movie catalogue"))
                    st.write(row.get("Overview") or "Synopsis unavailable")
                    if row["Source"] == "TMDB new release":
                        st.caption(f"TMDB community score: {row['TMDB Score']:.1f}/10. Not a personal prediction.")
                key = "visitor_" + scope + row["Key"]
                if rate:
                    score = st.select_slider("My rating", [i / 2 for i in range(1, 11)],
                                             value=profile.ratings.get(row["Key"], 4.0), key=key + "rating")
                    if st.button("Save rating", key=key + "save"):
                        profile.ratings[row["Key"]] = float(score)
                        profile.revision += 1
                        st.rerun()
                if st.button("Not interested", key=key + "hide"):
                    profile.blocked.add(row["Key"])
                    profile.revision += 1
                    st.rerun()


view = st.radio("View", ["For you", "Explore & rate", "My ratings", "New movies for you", "Preferences"],
                horizontal=True, key="visitor_view", label_visibility="collapsed")
if view in {"For you", "New movies for you"}:
    new_only = view == "New movies for you"
    st.header("New movies for you" if new_only else "Recommended for you")
    if not profile.ratings:
        st.info("Choose favourite genres or rate films in Explore & rate. Without ratings, suggestions use genre preferences or community popularity.")
    if new_only:
        st.caption("Content-based discovery. These are not collaborative-filtering rating predictions.")
    with st.spinner("Finding films for you..."):
        result = runtime.recommend(profile, algorithm, top_n, new_only=new_only)
    cards(result, "new" if new_only else "recommend")
elif view == "Explore & rate":
    query = st.text_input("Search movies", key="visitor_query")
    genre = st.selectbox("Genre", ["All", *catalog_genres(runtime.catalog)], key="visitor_genre")
    filtered = filter_catalog(runtime.catalog, query, genre).sort_values(["Ratings", "Title"], ascending=[False, True])
    pages = max(1, (len(filtered) + 8) // 9)
    selection = (query.strip(), genre)
    if st.session_state.get("visitor_selection") != selection or st.session_state.get("visitor_page", 1) > pages:
        st.session_state["visitor_page"] = 1
    st.session_state["visitor_selection"] = selection
    page = st.number_input("Page", min_value=1, max_value=pages, key="visitor_page")
    st.caption(f"{len(filtered):,} movies")
    cards(filtered.iloc[(page - 1) * 9:page * 9], "search", rate=True)
elif view == "My ratings":
    keys = [key for key in profile.ratings if key in runtime.catalog.index]
    st.header("My ratings")
    cards(runtime.catalog.loc[keys].head(30), "history", rate=True)
else:
    with st.form("visitor_preferences"):
        genres = st.multiselect("Favourite genres", catalog_genres(runtime.catalog), default=profile.genres)
        if st.form_submit_button("Save preferences", type="primary"):
            profile.genres = genres
            profile.revision += 1
            st.success("Preferences saved")
    for key in sorted(profile.blocked):
        if key in runtime.catalog.index:
            st.write(runtime.catalog.loc[key, "Title"])
            if st.button("Undo not interested", key="visitor_undo_" + key):
                profile.blocked.discard(key)
                profile.revision += 1
                st.rerun()

st.divider()
st.caption("Educational research demonstration. MovieLens data: GroupLens, University of Minnesota. No endorsement is implied.")
st.caption("This product uses the TMDB API but is not endorsed or certified by TMDB. Film artwork belongs to its respective owners.")
