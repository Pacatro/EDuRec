from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from edurec import settings

SYNTHETIC_DATASET = "synthetic"
SYNTHETIC_REQUIRED_FILES: tuple[str, ...] = (
    "users.csv",
    "items.csv",
    "interactions.csv",
)
DEFAULT_SEED = 42
DEFAULT_NUM_USERS = 8_000
DEFAULT_NUM_ITEMS = 1_200
DEFAULT_TARGET_INTERACTIONS = 100_000
LIST_SEPARATOR = " | "

# ``title`` is included as text and also used as the reference target for the
# ``prerequisites`` list, mirroring the DORIS schema convention.
SYNTHETIC_SCHEMA: dict[str, dict[str, Any]] = {
    "users": {
        "bin": ["gender"],
        "num": [
            "num_courses_completed",
            "num_certificates",
            "weekly_study_hours",
            "account_age_days",
            "avg_session_minutes",
        ],
        "cat": [
            "age_group",
            "country",
            "education_level",
            "employment_status",
            "study_field",
            "primary_device",
        ],
        "text": ["bio", "learning_goals"],
        "list": ["interests", "preferred_languages"],
    },
    "items": {
        "num": [
            "duration_minutes",
            "num_lessons",
            "num_quizzes",
            "avg_rating",
            "num_reviews",
            "created_year",
            "popularity",
        ],
        "cat": [
            "category",
            "subcategory",
            "difficulty",
            "language",
            "instructor",
            "content_type",
        ],
        "text": ["title", "description", "syllabus"],
        "list": ["tags", "skills", "prerequisites"],
        "refs": {"prerequisites": "title"},
        "cooc": [("subcategory", "category")],
    },
    "inter": {
        "bin": [],
        "num": ["watch_percentage", "time_spent_minutes", "quiz_score"],
        "cat": [
            "device",
            "completion_status",
            "recommendation_source",
            "session_period",
        ],
        "text": ["review_title", "review_text"],
        "list": ["topics_mentioned"],
    },
}

_CATALOG: dict[str, list[str]] = {
    "Data Science": [
        "Python",
        "Statistics",
        "Machine Learning",
        "Deep Learning",
        "Data Visualization",
        "SQL",
    ],
    "Web Development": [
        "JavaScript",
        "React",
        "HTML and CSS",
        "Node.js",
        "REST APIs",
        "Web Accessibility",
    ],
    "Business": [
        "Digital Marketing",
        "Finance",
        "Project Management",
        "Entrepreneurship",
        "Excel",
    ],
    "Design": [
        "UI/UX Design",
        "Graphic Design",
        "Figma",
        "Illustration",
        "Design Systems",
    ],
    "Languages": [
        "English",
        "Spanish",
        "French",
        "German",
        "Portuguese",
    ],
    "Cloud and DevOps": [
        "AWS",
        "Docker",
        "Kubernetes",
        "CI/CD",
        "Linux",
    ],
    "Personal Development": [
        "Productivity",
        "Communication",
        "Leadership",
        "Time Management",
        "Critical Thinking",
    ],
    "Mathematics": [
        "Algebra",
        "Calculus",
        "Discrete Mathematics",
        "Linear Algebra",
        "Probability",
    ],
    "Health and Wellness": [
        "Nutrition",
        "Fitness",
        "Mental Health",
        "Sleep Science",
    ],
    "Cybersecurity": [
        "Ethical Hacking",
        "Network Security",
        "Cryptography",
        "Incident Response",
    ],
}

_CATEGORIES: list[str] = list(_CATALOG)
_DIFFICULTIES: list[str] = ["Beginner", "Intermediate", "Advanced", "Expert"]
_CONTENT_TYPES: list[str] = [
    "course",
    "tutorial",
    "workshop",
    "webinar",
    "micro-course",
    "specialization",
]
_LANGUAGES: list[str] = ["en", "es", "fr", "de", "pt"]
_AGE_GROUPS: list[str] = ["18-24", "25-34", "35-44", "45-54", "55+"]
_COUNTRIES: list[str] = [
    "Spain",
    "France",
    "Germany",
    "Portugal",
    "United Kingdom",
    "Italy",
    "Netherlands",
    "Mexico",
    "Argentina",
    "Colombia",
    "Chile",
    "Peru",
    "Brazil",
    "United States",
    "Canada",
    "Morocco",
    "Egypt",
    "India",
    "China",
    "Japan",
    "Australia",
]
_EDUCATION_LEVELS: list[str] = [
    "High school",
    "Undergraduate",
    "Bachelor",
    "Master",
    "PhD",
]
_EMPLOYMENT: list[str] = [
    "Student",
    "Full-time employee",
    "Part-time employee",
    "Freelancer",
    "Unemployed",
    "Retired",
]
_DEVICES: list[str] = ["laptop", "desktop", "tablet", "smartphone"]
_RECOMMENDATION_SOURCES: list[str] = [
    "search",
    "recommendation",
    "category",
    "trending",
    "instructor",
    "newsletter",
]
_SESSION_PERIODS: list[str] = ["morning", "afternoon", "evening", "night"]
_INSTRUCTORS: list[str] = [
    "A. García",
    "L. Martin",
    "M. Rossi",
    "S. Fernández",
    "J. Dubois",
    "K. Müller",
    "P. Silva",
    "R. Johnson",
    "T. Nakamura",
    "C. Oliveira",
    "D. Novak",
    "E. Haddad",
    "F. Costa",
    "G. Petrov",
    "H. Andersen",
    "I. Kowalski",
    "N. Sharma",
    "O. Farouk",
    "V. Petrova",
    "W. Zhang",
]
_ROLES: list[str] = [
    "student",
    "junior developer",
    "data analyst",
    "product manager",
    "designer",
    "teacher",
    "researcher",
    "marketer",
    "engineer",
    "consultant",
    "entrepreneur",
]
_KEYWORDS: list[str] = [
    "analytics",
    "automation",
    "best practices",
    "cloud",
    "collaboration",
    "communication",
    "data",
    "debugging",
    "deployment",
    "design patterns",
    "documentation",
    "ethics",
    "frameworks",
    "git",
    "hands-on",
    "iteration",
    "machine learning",
    "metrics",
    "modeling",
    "networking",
    "optimization",
    "pipelines",
    "presentation",
    "problem solving",
    "productivity",
    "prototyping",
    "reporting",
    "research",
    "scalability",
    "security",
    "storytelling",
    "strategy",
    "testing",
    "visualization",
    "workflow",
    "writing",
]

_TITLE_TEMPLATES: list[str] = [
    "{subcat} Fundamentals",
    "Complete {subcat} Bootcamp",
    "{subcat} for Beginners",
    "Advanced {subcat}",
    "Practical {subcat} Projects",
    "{subcat}: From Zero to Hero",
    "Hands-On {subcat}",
    "{subcat} Essentials",
    "{subcat} Masterclass",
    "Introduction to {subcat}",
    "{subcat} in Practice",
    "Building Real Projects with {subcat}",
    "{subcat} Deep Dive",
    "{subcat} Crash Course",
    "{subcat} for Professionals",
]
_DESCRIPTION_TEMPLATES: list[str] = [
    (
        "A {difficulty_lower} {ctype} that introduces {subcat} within the broader "
        "{category} track. You will work through guided examples and finish with a "
        "portfolio-ready project."
    ),
    (
        "This {ctype} covers the core ideas of {subcat} step by step. Each lesson "
        "combines short theory videos, readings and graded exercises so you can "
        "apply {category} concepts immediately."
    ),
    (
        "Learn {subcat} with a practical, project-based approach. The {ctype} "
        "starts from the foundations and gradually moves to realistic {category} "
        "scenarios and common pitfalls."
    ),
    (
        "An in-depth {ctype} on {subcat}. You will build several mini-projects, "
        "review industry {keywords} and receive feedback on your solutions."
    ),
    (
        "Designed for {audience}, this {ctype} explains how {subcat} fits into "
        "modern {category} workflows, with tips on tools, {keyword} and best "
        "practices."
    ),
]
_SYLLABUS_MODULE_TEMPLATES: list[str] = [
    "Getting started with {subcat} and setting up the environment.",
    "Core concepts of {subcat} explained with worked examples.",
    "Hands-on lab: applying {keyword} to a realistic dataset or scenario.",
    "Common mistakes, debugging strategies and {keyword}.",
    (
        "Scaling your {subcat} solution and integrating it with the wider "
        "{category} stack."
    ),
    "Capstone project: designing and presenting an end-to-end solution.",
    "Assessments, code review and next steps for continued learning.",
]

_BIO_TEMPLATES: list[str] = [
    (
        "I am a {age} {role} from {country}. I work with {interest} and I am "
        "currently strengthening my skills through short online courses. I prefer "
        "practical, project-based material and I usually study in the {period}."
    ),
    (
        "{role_cap} based in {country}, aged {age}. I moved into {interest} a "
        "couple of years ago and use online learning to keep up with new tools. I "
        "value clear explanations and real examples."
    ),
    (
        "Curious learner from {country} with a background in {education}. My main "
        "interests are {interest}. I like bite-sized lessons that fit around my "
        "{employment_lower} schedule."
    ),
    (
        "I am a {age} {role} who enjoys {interest}. I combine self-study with "
        "hands-on projects and I am looking for structured paths in {category}."
    ),
]
_GOAL_TEMPLATES: list[str] = [
    (
        "Complete a structured {category} path and earn at least two certificates "
        "this year."
    ),
    "Improve my {interest} skills enough to apply for a more technical role.",
    "Build a portfolio project that demonstrates {keyword} and {keyword}.",
    "Prepare for a certification exam in {category} while balancing work.",
    (
        "Understand the fundamentals of {category} to collaborate better with "
        "specialist teams."
    ),
    "Stay current with {keyword} and {keyword} through short weekly sessions.",
]

_POSITIVE_TITLES: list[str] = [
    "Excellent introduction to {subcat}",
    "Loved this course",
    "Very clear and practical",
    "Best {subcat} resource so far",
    "Highly recommended",
    "Great balance of theory and practice",
]
_NEUTRAL_TITLES: list[str] = [
    "Good but could go deeper",
    "Solid overview",
    "Decent for the price",
    "Useful with some caveats",
    "Okay introduction",
]
_NEGATIVE_TITLES: list[str] = [
    "Not what I expected",
    "Too basic for me",
    "Confusing explanations",
    "Outdated material",
    "Needs better examples",
]
_POSITIVE_BODIES: list[str] = [
    (
        "The {subcat} modules were well structured and the exercises really helped "
        "me consolidate the concepts. The instructor answered questions quickly."
    ),
    (
        "I came in with almost no background and finished feeling confident. The "
        "examples around {keyword} were especially useful."
    ),
    (
        "Great pace, clear slides and a capstone that I could actually add to my "
        "portfolio. I would take another {category} course from this instructor."
    ),
    (
        "The mix of short videos and hands-on labs kept me engaged. The section on "
        "{keyword} tied everything together."
    ),
    (
        "Exactly the practical depth I was looking for. The {subcat} project "
        "walkthrough alone was worth it."
    ),
]
_NEUTRAL_BODIES: list[str] = [
    (
        "The content is correct and the structure is fine, but some {subcat} "
        "topics are only skimmed. It works as a starting point."
    ),
    (
        "Good production quality, although the {keyword} section felt a bit "
        "rushed. I had to consult extra documentation."
    ),
    (
        "Useful overview of {subcat}. It did not cover advanced {category} "
        "material, so I would pair it with a deeper resource."
    ),
    (
        "The course met my basic expectations. The quizzes are easy and the "
        "examples could be more varied."
    ),
]
_NEGATIVE_BODIES: list[str] = [
    (
        "The {subcat} explanations were hard to follow and the audio quality was "
        "inconsistent. I stopped halfway through."
    ),
    (
        "Many examples are outdated and no longer match current {category} tools. "
        "The promised {keyword} project never goes beyond a sketch."
    ),
    (
        "Too much theory and almost no practice. I expected hands-on {subcat} "
        "exercises but found mostly slides."
    ),
    (
        "The level was far below what the title suggests. Not recommended if you "
        "already know the basics of {category}."
    ),
]


def _pick(rng: np.random.Generator, values: list[str]) -> str:
    return values[int(rng.integers(len(values)))]


def _join_tokens(tokens: list[str]) -> str:
    return LIST_SEPARATOR.join(dict.fromkeys(token for token in tokens if token))


def _build_items(
    rng: np.random.Generator,
    num_items: int,
) -> pd.DataFrame:
    pairs = [
        (category, subcategory)
        for category, subcategories in _CATALOG.items()
        for subcategory in subcategories
    ]
    audience = ["beginners", "intermediate learners", "professionals", "teams"]
    difficulty_probs = np.array([0.35, 0.35, 0.22, 0.08])

    rows: list[dict[str, Any]] = []
    for index in range(num_items):
        category, subcategory = pairs[index % len(pairs)]
        difficulty = _DIFFICULTIES[
            int(rng.choice(len(_DIFFICULTIES), p=difficulty_probs))
        ]
        content_type = _pick(rng, _CONTENT_TYPES)
        language = _pick(rng, _LANGUAGES)
        instructor = _pick(rng, _INSTRUCTORS)
        audience_value = _pick(rng, audience)
        keyword_a, keyword_b = (
            _pick(rng, _KEYWORDS),
            _pick(rng, _KEYWORDS),
        )
        tags = [subcategory.lower(), category.lower(), keyword_a, keyword_b]
        tags.extend(_KEYWORDS[int(i)] for i in rng.integers(0, len(_KEYWORDS), 3))
        skills = [
            f"{subcategory} basics",
            f"{subcategory} applied to {category.lower()}",
            keyword_a,
            keyword_b,
        ]

        duration = int(np.clip(rng.normal(240, 160), 30, 1_200) // 5 * 5)
        num_lessons = int(np.clip(duration / rng.uniform(8, 20), 4, 120))
        num_quizzes = int(np.clip(num_lessons / rng.uniform(2.0, 5.0), 1, 40))

        title = _pick(rng, _TITLE_TEMPLATES).format(subcat=subcategory)
        if rng.random() < 0.25:
            title = f"{title} ({_pick(rng, _LANGUAGES).upper()})"

        description = _pick(rng, _DESCRIPTION_TEMPLATES).format(
            difficulty_lower=difficulty.lower(),
            ctype=content_type,
            subcat=subcategory,
            category=category,
            keywords=keyword_a,
            keyword=keyword_b,
            audience=audience_value,
        )

        num_modules = int(rng.integers(4, 7))
        module_indexes = rng.choice(
            len(_SYLLABUS_MODULE_TEMPLATES), size=num_modules, replace=False
        )
        modules = [
            "Module {}: {}".format(
                position,
                _SYLLABUS_MODULE_TEMPLATES[int(module_index)].format(
                    subcat=subcategory,
                    category=category,
                    keyword=_pick(rng, _KEYWORDS),
                ),
            )
            for position, module_index in enumerate(module_indexes, start=1)
        ]
        syllabus = " ".join(modules)

        rows.append(
            {
                "item_id": index + 1,
                "title": title,
                "description": description,
                "syllabus": syllabus,
                "category": category,
                "subcategory": subcategory,
                "difficulty": difficulty,
                "language": language,
                "instructor": instructor,
                "content_type": content_type,
                "duration_minutes": duration,
                "num_lessons": num_lessons,
                "num_quizzes": num_quizzes,
                "avg_rating": round(float(rng.uniform(3.0, 4.9)), 2),
                "num_reviews": int(np.clip(rng.lognormal(4.5, 1.0), 5, 5_000)),
                "created_year": int(rng.integers(2017, 2025)),
                "popularity": int(np.clip(rng.lognormal(7.0, 1.2), 50, 200_000)),
                "tags": _join_tokens(tags),
                "skills": _join_tokens(skills),
                "prerequisites": "",
            }
        )

    items = pd.DataFrame(rows)

    # Advanced items may declare prerequisites drawn from easier items of the
    # same subcategory. Titles are reused as reference keys by the KG builder.
    difficulties_list = items["difficulty"].tolist()
    subcategories_list = items["subcategory"].tolist()
    titles_list = items["title"].tolist()
    prerequisites_column: list[str] = [""] * len(items)
    titles_by_subcategory: dict[str, list[str]] = {}
    for index, (difficulty, subcategory, title) in enumerate(
        zip(difficulties_list, subcategories_list, titles_list, strict=True)
    ):
        if difficulty in {"Advanced", "Expert"}:
            candidates = titles_by_subcategory.get(subcategory, [])
            if candidates:
                sample_size = int(min(len(candidates), rng.integers(1, 3)))
                chosen = rng.choice(len(candidates), size=sample_size, replace=False)
                prerequisites_column[index] = _join_tokens(
                    [candidates[int(i)] for i in chosen]
                )
        titles_by_subcategory.setdefault(subcategory, []).append(title)

    items["prerequisites"] = prerequisites_column

    return items


def _build_users(
    rng: np.random.Generator,
    num_users: int,
) -> pd.DataFrame:
    level_by_education = {
        "High school": 0,
        "Undergraduate": 0,
        "Bachelor": 1,
        "Master": 2,
        "PhD": 3,
    }
    rows: list[dict[str, Any]] = []
    for index in range(num_users):
        age_group = _pick(rng, _AGE_GROUPS)
        country = _pick(rng, _COUNTRIES)
        education = _pick(rng, _EDUCATION_LEVELS)
        employment = _pick(rng, _EMPLOYMENT)
        employment_lower = employment.lower()
        device = _pick(rng, _DEVICES)
        period = _pick(rng, _SESSION_PERIODS)
        role = _pick(rng, _ROLES)
        primary_category = _pick(rng, _CATEGORIES)
        interest_categories = rng.choice(
            len(_CATEGORIES),
            size=int(rng.integers(2, 5)),
            replace=False,
        )
        interests = [_CATEGORIES[int(i)] for i in interest_categories]
        interest_text = " and ".join(interests[:2])
        keyword_a, keyword_b = _pick(rng, _KEYWORDS), _pick(rng, _KEYWORDS)

        bio = _pick(rng, _BIO_TEMPLATES).format(
            age=age_group,
            role=role,
            role_cap=role.capitalize(),
            country=country,
            interest=interest_text,
            period=period,
            education=education,
            employment_lower=employment_lower,
            category=primary_category,
        )
        goals = _pick(rng, _GOAL_TEMPLATES).format(
            interest=interest_text,
            keyword=keyword_a,
            keyword2=keyword_b,
            category=primary_category,
        )
        if rng.random() < 0.5:
            goals += " " + _pick(rng, _GOAL_TEMPLATES).format(
                interest=interest_text,
                keyword=keyword_b,
                keyword2=keyword_a,
                category=primary_category,
            )

        languages = list(
            rng.choice(_LANGUAGES, size=int(rng.integers(1, 4)), replace=False)
        )

        rows.append(
            {
                "user_id": index + 1,
                "gender": int(rng.integers(0, 2)),
                "age_group": age_group,
                "country": country,
                "education_level": education,
                "employment_status": employment,
                "study_field": primary_category,
                "level": level_by_education[education],
                "primary_device": device,
                "num_courses_completed": int(np.clip(rng.poisson(6), 0, 60)),
                "num_certificates": int(np.clip(rng.poisson(2), 0, 25)),
                "weekly_study_hours": round(
                    float(np.clip(rng.normal(6, 3), 0.5, 30)), 1
                ),
                "account_age_days": int(np.clip(rng.integers(30, 1_460), 30, 1_460)),
                "avg_session_minutes": int(np.clip(rng.normal(42, 18), 5, 180)),
                "bio": bio,
                "learning_goals": goals,
                "interests": _join_tokens(interests),
                "preferred_languages": _join_tokens(list(languages)),
                "interest_ids": interest_categories.tolist(),
            }
        )

    return pd.DataFrame(rows)


def _build_interactions(
    rng: np.random.Generator,
    users: pd.DataFrame,
    items: pd.DataFrame,
    target_interactions: int,
) -> pd.DataFrame:
    item_categories = items["category"].to_numpy()
    subcategories = items["subcategory"].to_numpy()
    difficulties = items["difficulty"].to_numpy()
    diff_index = {value: index for index, value in enumerate(_DIFFICULTIES)}
    item_diff = np.array([diff_index[value] for value in difficulties])
    popularity = items["popularity"].to_numpy(dtype=np.float64)
    pop_weight = popularity / popularity.sum()
    tags = items["tags"].to_numpy()

    category_index = {value: index for index, value in enumerate(_CATEGORIES)}
    mean_per_user = target_interactions / len(users)

    item_cat_idx = np.array([category_index[value] for value in item_categories])
    duration_by_item = items["duration_minutes"].to_numpy()
    all_tags = [str(value).split(LIST_SEPARATOR) for value in tags]

    user_ids: list[int] = []
    item_ids: list[int] = []
    timestamps: list[int] = []
    ratings: list[int] = []
    watch: list[int] = []
    time_spent: list[int] = []
    quiz_scores: list[float] = []
    devices: list[str] = []
    completion: list[str] = []
    sources: list[str] = []
    periods: list[str] = []
    review_titles: list[str] = []
    review_texts: list[str] = []
    topics: list[str] = []

    user_id_values = users["user_id"].to_numpy()
    interest_values = users["interest_ids"].tolist()
    level_values = users["level"].to_numpy()
    device_values = users["primary_device"].tolist()

    for user_index in range(len(users)):
        interest_ids = np.asarray(interest_values[user_index], dtype=int)
        interest_mask = np.isin(item_cat_idx, interest_ids)
        level = int(level_values[user_index])
        base = int(np.clip(rng.poisson(mean_per_user), 6, 160))
        if base >= len(items):
            base = len(items) - 1

        # Gumbel top-k weighted sampling without replacement.
        weights = pop_weight * (1.0 + 3.5 * interest_mask)
        weights *= np.exp(-0.7 * np.abs(item_diff - level))
        gumbel = rng.gumbel(size=len(items))
        scores = np.log(weights) + gumbel
        chosen = np.argpartition(scores, -base)[-base:]

        start = int(
            pd.Timestamp("2021-01-01").timestamp()
            + rng.integers(0, 24 * 3600 * 365 * 2)
        )
        gaps = np.clip(rng.lognormal(mean=2.2, sigma=0.8, size=base), 1, 720).astype(
            int
        )
        cumulative = start + np.cumsum(gaps * 3600)
        user_bias = float(rng.normal(0.0, 0.45))

        order = np.argsort(cumulative)
        for position in order:
            item_index = int(chosen[position])
            is_interest = bool(interest_mask[item_index])
            diff_match = float(np.exp(-0.7 * abs(item_diff[item_index] - level)))

            similarity = 2.7 + 1.15 * is_interest + 0.75 * diff_match
            rating_value = int(
                np.clip(
                    np.rint(similarity + user_bias + rng.normal(0.0, 0.65)),
                    1,
                    5,
                )
            )
            watch_value = int(
                np.clip(38 + 13 * rating_value + rng.normal(0, 12), 3, 100)
            )
            if watch_value >= 95:
                completion_value = "completed"
            elif watch_value >= 55:
                completion_value = "in_progress"
            else:
                completion_value = "dropped"

            duration = int(duration_by_item[item_index])
            spent = max(
                1, int(duration * watch_value / 100 * float(rng.uniform(0.75, 1.25)))
            )
            if completion_value == "completed":
                quiz_value = float(
                    np.clip(45 + 11 * rating_value + rng.normal(0, 8), 0, 100)
                )
            else:
                quiz_value = float("nan")

            subcategory = subcategories[item_index]
            category = item_categories[item_index]
            if rating_value >= 4:
                title_bucket, body_bucket = _POSITIVE_TITLES, _POSITIVE_BODIES
            elif rating_value == 3:
                title_bucket, body_bucket = _NEUTRAL_TITLES, _NEUTRAL_BODIES
            else:
                title_bucket, body_bucket = _NEGATIVE_TITLES, _NEGATIVE_BODIES

            review_title = _pick(rng, title_bucket).format(subcat=subcategory)
            keyword = _pick(rng, _KEYWORDS)
            review_text = _pick(rng, body_bucket).format(
                subcat=subcategory,
                category=category,
                keyword=keyword,
            )
            if rng.random() < 0.4:
                review_text += " " + _pick(rng, body_bucket).format(
                    subcat=subcategory,
                    category=category,
                    keyword=_pick(rng, _KEYWORDS),
                )

            item_tags = all_tags[item_index]
            topic_sample = rng.choice(
                len(item_tags),
                size=int(min(len(item_tags), rng.integers(2, 5))),
                replace=False,
            )
            topic_tokens = [item_tags[int(i)] for i in topic_sample]

            user_ids.append(int(user_id_values[user_index]))
            item_ids.append(item_index + 1)
            timestamps.append(int(cumulative[position]))
            ratings.append(rating_value)
            watch.append(watch_value)
            time_spent.append(spent)
            quiz_scores.append(quiz_value)
            devices.append(
                device_values[user_index]
                if rng.random() < 0.75
                else _pick(rng, _DEVICES)
            )
            completion.append(completion_value)
            sources.append(_pick(rng, _RECOMMENDATION_SOURCES))
            periods.append(_pick(rng, _SESSION_PERIODS))
            review_titles.append(review_title)
            review_texts.append(review_text)
            topics.append(_join_tokens(topic_tokens))

    interactions = pd.DataFrame(
        {
            "user_id": user_ids,
            "item_id": item_ids,
            "created_at": pd.to_datetime(timestamps, unit="s", utc=True),
            "rating": ratings,
            "watch_percentage": watch,
            "time_spent_minutes": time_spent,
            "quiz_score": quiz_scores,
            "device": devices,
            "completion_status": completion,
            "recommendation_source": sources,
            "session_period": periods,
            "review_title": review_titles,
            "review_text": review_texts,
            "topics_mentioned": topics,
        }
    )
    return interactions


def generate_synthetic_raw(
    output_dir: Path | str | None = None,
    *,
    num_users: int = DEFAULT_NUM_USERS,
    num_items: int = DEFAULT_NUM_ITEMS,
    target_interactions: int = DEFAULT_TARGET_INTERACTIONS,
    seed: int = DEFAULT_SEED,
    force: bool = False,
) -> Path:
    """Generate the synthetic dataset CSVs and return the output folder.

    Args:
        output_dir: Destination folder. Defaults to
            ``data/raw/synthetic`` relative to the project root.
        num_users: Number of user profiles to synthesise.
        num_items: Number of course-like items to synthesise.
        target_interactions: Approximate number of interactions to generate.
            The final count varies because each user draws a variable number of
            items.
        seed: Random seed for full reproducibility.
        force: Regenerate the files even if they already exist.

    Returns:
        The folder containing ``users.csv``, ``items.csv`` and
        ``interactions.csv``.
    """

    folder = (
        Path(output_dir)
        if output_dir is not None
        else settings.RAW_DATA_FOLDER / SYNTHETIC_DATASET
    )
    folder.mkdir(parents=True, exist_ok=True)

    if not force and all((folder / name).exists() for name in SYNTHETIC_REQUIRED_FILES):
        return folder

    rng = np.random.default_rng(seed)

    items = _build_items(rng, num_items)
    users = _build_users(rng, num_users)
    interactions = _build_interactions(rng, users, items, target_interactions)

    users = users.drop(columns=["level", "interest_ids"])

    users.to_csv(folder / "users.csv", index=False)
    items.to_csv(folder / "items.csv", index=False)
    interactions.to_csv(folder / "interactions.csv", index=False)
    (folder / "README.md").write_text(
        _data_card(len(users), len(items), len(interactions))
    )

    return folder


def _data_card(num_users: int, num_items: int, num_interactions: int) -> str:
    return f"""# Synthetic EDuRec dataset

Deterministic synthetic e-learning dataset generated by
`edurec.datasets.synthetic.generate_synthetic_raw`.

- Users: {num_users:,}
- Items: {num_items:,}
- Interactions: {num_interactions:,}

Columns

- `users.csv`: `user_id`, `gender`, `age_group`, `country`,
  `education_level`, `employment_status`, `study_field`, `primary_device`,
  `num_courses_completed`, `num_certificates`, `weekly_study_hours`,
  `account_age_days`, `avg_session_minutes`, `bio`, `learning_goals`,
  `interests`, `preferred_languages`.
- `items.csv`: `item_id`, `title`, `description`, `syllabus`, `category`,
  `subcategory`, `difficulty`, `language`, `instructor`, `content_type`,
  `duration_minutes`, `num_lessons`, `num_quizzes`, `avg_rating`,
  `num_reviews`, `created_year`, `popularity`, `tags`, `skills`,
  `prerequisites`.
- `interactions.csv`: `user_id`, `item_id`, `created_at`, `rating`,
  `watch_percentage`, `time_spent_minutes`, `quiz_score`, `device`,
  `completion_status`, `recommendation_source`, `session_period`,
  `review_title`, `review_text`, `topics_mentioned`.

List-valued columns use `{LIST_SEPARATOR.strip()}` as separator.
"""
