import io
import json
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

import streamlit as st
from pypdf import PdfReader

from agent import analyze_job


APP_DATA_DIR = Path(__file__).parent / ".applyai"
LAST_RESUME_FILE = APP_DATA_DIR / "last_resume.bin"
LAST_RESUME_METADATA = APP_DATA_DIR / "last_resume.json"
CANDIDATE_METADATA_FILE = APP_DATA_DIR / "candidate_metadata.json"
CANDIDATE_METADATA_SCHEMA_VERSION = 2
ANALYSIS_CACHE_VERSION = "candidate-settings-v2"

WORK_ARRANGEMENT_OPTIONS = {
    "Remote": "remote",
    "Hybrid": "hybrid",
    "Onsite": "onsite"
}
JOB_TYPE_OPTIONS = {
    "Full-time": "full_time",
    "Part-time": "part_time",
    "Internship": "internship",
    "Contract": "contract"
}
JOB_AREA_OPTIONS = {
    "Software Engineering": "software_engineering",
    "Data Engineering": "data_engineering",
    "Data Analysis": "data_analysis",
    "IT / Support": "it_support",
    "AI / ML": "ai_ml"
}
CLEARANCE_OPTIONS = {
    "Not provided": "unknown",
    "None": "none",
    "Eligible": "eligible",
    "Active": "active"
}


# -------------------------
# Page Setup
# -------------------------

st.set_page_config(
    page_title="ApplyAI",
    page_icon="💼",
    layout="centered"
)

st.title("💼 ApplyAI")

st.write(
    "Paste a job description below. "
    "ApplyAI will compare the position "
    "against an uploaded resume."
)


# -------------------------
# Helper
# -------------------------

def clean_text(text):

    lines = []

    for line in text.splitlines():

        line = line.strip()

        if line:
            lines.append(line)

    return "\n".join(lines)


def extract_resume_text(file_data, filename):

    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        reader = PdfReader(io.BytesIO(file_data))
        text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    elif extension == ".docx":
        with zipfile.ZipFile(io.BytesIO(file_data)) as archive:
            document_xml = archive.read(
                "word/document.xml"
            )

        root = ElementTree.fromstring(document_xml)
        namespace = (
            "{http://schemas.openxmlformats.org/"
            "wordprocessingml/2006/main}"
        )
        paragraphs = []

        for paragraph in root.iter(f"{namespace}p"):
            paragraph_text = "".join(
                node.text or ""
                for node in paragraph.iter(f"{namespace}t")
            ).strip()

            if paragraph_text:
                paragraphs.append(paragraph_text)

        text = "\n".join(paragraphs)

    elif extension == ".txt":
        try:
            text = file_data.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = file_data.decode("cp1252")

    else:
        raise ValueError(
            "Resume must be a PDF, DOCX, or TXT file."
        )

    text = clean_text(text)

    if not text:
        raise ValueError(
            "No readable text was found in the resume. "
            "Scanned PDFs need OCR before uploading."
        )

    return text


def get_last_resume_metadata():

    if (
        not LAST_RESUME_FILE.exists()
        or not LAST_RESUME_METADATA.exists()
    ):
        return None

    try:
        metadata = json.loads(
            LAST_RESUME_METADATA.read_text(
                encoding="utf-8"
            )
        )
        if not metadata.get("filename"):
            return None
        return metadata
    except (OSError, json.JSONDecodeError):
        return None


def save_last_resume(file_data, filename):

    APP_DATA_DIR.mkdir(exist_ok=True)

    metadata = {
        "filename": Path(filename).name,
        "saved_at": datetime.now(
            timezone.utc
        ).isoformat()
    }

    resume_temp = APP_DATA_DIR / "last_resume.bin.tmp"
    metadata_temp = APP_DATA_DIR / "last_resume.json.tmp"

    resume_temp.write_bytes(file_data)
    metadata_temp.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    resume_temp.replace(LAST_RESUME_FILE)
    metadata_temp.replace(LAST_RESUME_METADATA)


def load_last_resume():

    metadata = get_last_resume_metadata()

    if metadata is None:
        raise ValueError(
            "No previously uploaded resume is available."
        )

    return (
        LAST_RESUME_FILE.read_bytes(),
        metadata["filename"]
    )


def default_candidate_metadata():

    return {
        "schema_version": CANDIDATE_METADATA_SCHEMA_VERSION,
        "work_authorization": {
            "us_authorized": None,
            "requires_sponsorship": None
        },
        "location": {
            "current": None,
            "willing_to_relocate": None
        },
        "work_preferences": {
            "arrangements": [],
            "job_types": [],
            "preferred_job_areas": []
        },
        "availability": {
            "earliest_start_date": None
        },
        "education": {
            "graduation_date": None
        },
        "travel": {
            "willing": None,
            "maximum_percentage": None
        },
        "security_clearance": {
            "status": "unknown"
        },
        "compensation": {
            "minimum_amount": None,
            "currency": "USD",
            "period": "annual"
        }
    }


def normalize_optional_date(value):

    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, str):
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError:
            return None

    return None


def normalize_choice_list(values, allowed_values):

    if not isinstance(values, list):
        return []

    return [
        value
        for value in dict.fromkeys(values)
        if value in allowed_values
    ]


def normalize_candidate_metadata(metadata):
    """
    Preserve future metadata sections while validating known fields.
    """

    if isinstance(metadata, dict):
        normalized = dict(metadata)
    else:
        normalized = {}

    normalized["schema_version"] = (
        CANDIDATE_METADATA_SCHEMA_VERSION
    )

    authorization = normalized.get(
        "work_authorization"
    )

    if isinstance(authorization, dict):
        authorization = dict(authorization)
    else:
        authorization = {}

    for field in (
        "us_authorized",
        "requires_sponsorship"
    ):
        value = authorization.get(field)
        authorization[field] = (
            value
            if isinstance(value, bool)
            else None
        )

    normalized["work_authorization"] = authorization

    location = normalized.get("location")
    location = dict(location) if isinstance(location, dict) else {}
    current_location = location.get("current")
    location["current"] = (
        current_location.strip()
        if isinstance(current_location, str)
        and current_location.strip()
        else None
    )
    willing_to_relocate = location.get(
        "willing_to_relocate"
    )
    location["willing_to_relocate"] = (
        willing_to_relocate
        if isinstance(willing_to_relocate, bool)
        else None
    )
    normalized["location"] = location

    work_preferences = normalized.get("work_preferences")
    work_preferences = (
        dict(work_preferences)
        if isinstance(work_preferences, dict)
        else {}
    )
    work_preferences["arrangements"] = normalize_choice_list(
        work_preferences.get("arrangements"),
        set(WORK_ARRANGEMENT_OPTIONS.values())
    )
    work_preferences["job_types"] = normalize_choice_list(
        work_preferences.get("job_types"),
        set(JOB_TYPE_OPTIONS.values())
    )
    work_preferences[
        "preferred_job_areas"
    ] = normalize_choice_list(
        work_preferences.get("preferred_job_areas"),
        set(JOB_AREA_OPTIONS.values())
    )
    normalized["work_preferences"] = work_preferences

    availability = normalized.get("availability")
    availability = (
        dict(availability)
        if isinstance(availability, dict)
        else {}
    )
    availability[
        "earliest_start_date"
    ] = normalize_optional_date(
        availability.get("earliest_start_date")
    )
    normalized["availability"] = availability

    education = normalized.get("education")
    education = (
        dict(education)
        if isinstance(education, dict)
        else {}
    )
    education["graduation_date"] = normalize_optional_date(
        education.get("graduation_date")
    )
    normalized["education"] = education

    travel = normalized.get("travel")
    travel = dict(travel) if isinstance(travel, dict) else {}
    willing_to_travel = travel.get("willing")
    travel["willing"] = (
        willing_to_travel
        if isinstance(willing_to_travel, bool)
        else None
    )
    maximum_travel = travel.get("maximum_percentage")
    travel["maximum_percentage"] = (
        int(maximum_travel)
        if isinstance(maximum_travel, (int, float))
        and not isinstance(maximum_travel, bool)
        and 0 <= maximum_travel <= 100
        else None
    )
    normalized["travel"] = travel

    security_clearance = normalized.get("security_clearance")
    security_clearance = (
        dict(security_clearance)
        if isinstance(security_clearance, dict)
        else {}
    )
    clearance_status = security_clearance.get("status")
    security_clearance["status"] = (
        clearance_status
        if clearance_status in {
            "none", "eligible", "active", "unknown"
        }
        else "unknown"
    )
    normalized["security_clearance"] = security_clearance

    compensation = normalized.get("compensation")
    compensation = (
        dict(compensation)
        if isinstance(compensation, dict)
        else {}
    )
    minimum_amount = compensation.get("minimum_amount")
    compensation["minimum_amount"] = (
        int(minimum_amount)
        if isinstance(minimum_amount, (int, float))
        and not isinstance(minimum_amount, bool)
        and minimum_amount >= 0
        else None
    )
    compensation["currency"] = "USD"
    compensation["period"] = "annual"
    normalized["compensation"] = compensation

    return normalized


def load_candidate_metadata():

    if not CANDIDATE_METADATA_FILE.exists():
        return default_candidate_metadata()

    try:
        metadata = json.loads(
            CANDIDATE_METADATA_FILE.read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError):
        return default_candidate_metadata()

    normalized = normalize_candidate_metadata(metadata)

    if normalized != metadata:
        try:
            save_candidate_metadata(normalized)
        except OSError:
            pass

    return normalized


def save_candidate_metadata(metadata):

    APP_DATA_DIR.mkdir(exist_ok=True)
    normalized = normalize_candidate_metadata(metadata)
    metadata_temp = (
        APP_DATA_DIR / "candidate_metadata.json.tmp"
    )

    metadata_temp.write_text(
        json.dumps(normalized, indent=2),
        encoding="utf-8"
    )
    metadata_temp.replace(CANDIDATE_METADATA_FILE)


def optional_boolean_index(value):

    if value is True:
        return 1

    if value is False:
        return 2

    return 0


def optional_boolean_value(label):

    if label == "Yes":
        return True

    if label == "No":
        return False

    return None


def option_labels_for_values(options, values):

    selected_values = set(values or [])

    return [
        label
        for label, value in options.items()
        if value in selected_values
    ]


def metadata_date_value(value):

    if not value:
        return None

    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


@st.cache_data
def cached_analysis(
    job_description,
    resume_text,
    candidate_metadata,
    cache_version
):

    return analyze_job(
        job_description,
        resume_text,
        candidate_metadata
    )


def status_icon(status):

    if status == "direct":
        return "✅"

    if status == "partial":
        return "⚠️"

    if status == "unknown":
        return "❔"

    return "❌"


def status_name(status):

    if status == "direct":
        return "Match"

    if status == "partial":
        return "Partial"

    if status == "unknown":
        return "Not provided"

    return "Missing"


def fit_status_icon(status):

    return {
        "match": "✅",
        "mismatch": "❌",
        "warning": "⚠️",
        "unknown": "❔",
        "info": "ℹ️"
    }.get(status, "ℹ️")


# -------------------------
# Resume
# -------------------------

st.subheader("Resume")

last_resume_metadata = get_last_resume_metadata()

resume_source = st.radio(
    "Resume source",
    (
        "Upload a resume",
        "Use last uploaded"
    ),
    index=(
        1
        if last_resume_metadata is not None
        else 0
    ),
    horizontal=True
)

uploaded_resume = None

if resume_source == "Upload a resume":
    uploaded_resume = st.file_uploader(
        "Upload resume",
        type=["pdf", "docx", "txt"],
        help="Supported formats: PDF, DOCX, and TXT."
    )

elif last_resume_metadata is None:
    st.warning(
        "No previously uploaded resume is available yet."
    )

else:
    st.info(
        "Using last uploaded resume: "
        f"{last_resume_metadata['filename']}"
    )

st.caption(
    "The last successful upload is stored locally "
    "and excluded from Git."
)


# -------------------------
# Candidate Profile / Settings
# -------------------------

stored_candidate_metadata = load_candidate_metadata()
optional_boolean_labels = (
    "Not provided",
    "Yes",
    "No"
)

with st.expander(
    "Candidate Profile / Job Preferences & Eligibility",
    expanded=True
):
    st.caption(
        "These candidate-supplied settings are stored locally. "
        "They are not inferred from your resume."
    )

    eligibility_tab, preferences_tab = st.tabs(
        ("Eligibility", "Job Preferences")
    )

    with eligibility_tab:
        stored_work_authorization = stored_candidate_metadata[
            "work_authorization"
        ]

        authorization_answer = st.selectbox(
            (
                "Are you currently authorized to work "
                "in the United States?"
            ),
            optional_boolean_labels,
            index=optional_boolean_index(
                stored_work_authorization["us_authorized"]
            ),
            key="us_authorized"
        )

        sponsorship_answer = st.selectbox(
            (
                "Will you now or in the future require employer "
                "sponsorship to work in the United States?"
            ),
            optional_boolean_labels,
            index=optional_boolean_index(
                stored_work_authorization[
                    "requires_sponsorship"
                ]
            ),
            key="requires_sponsorship"
        )

        stored_clearance = stored_candidate_metadata[
            "security_clearance"
        ]["status"]
        clearance_labels = list(CLEARANCE_OPTIONS)
        clearance_answer = st.selectbox(
            "Security clearance status",
            clearance_labels,
            index=clearance_labels.index(
                next(
                    label
                    for label, value in CLEARANCE_OPTIONS.items()
                    if value == stored_clearance
                )
            ),
            key="security_clearance"
        )

    with preferences_tab:
        stored_location = stored_candidate_metadata["location"]
        current_location = st.text_input(
            "Current location",
            value=stored_location["current"] or "",
            placeholder="City, state/province, country",
            key="current_location"
        )

        relocation_answer = st.selectbox(
            "Willing to relocate?",
            optional_boolean_labels,
            index=optional_boolean_index(
                stored_location["willing_to_relocate"]
            ),
            key="willing_to_relocate"
        )

        stored_preferences = stored_candidate_metadata[
            "work_preferences"
        ]
        arrangement_labels = st.multiselect(
            "Work arrangement preferences",
            list(WORK_ARRANGEMENT_OPTIONS),
            default=option_labels_for_values(
                WORK_ARRANGEMENT_OPTIONS,
                stored_preferences["arrangements"]
            ),
            key="work_arrangements"
        )
        job_type_labels = st.multiselect(
            "Job type preferences",
            list(JOB_TYPE_OPTIONS),
            default=option_labels_for_values(
                JOB_TYPE_OPTIONS,
                stored_preferences["job_types"]
            ),
            key="job_types"
        )
        job_area_labels = st.multiselect(
            "Preferred job areas",
            list(JOB_AREA_OPTIONS),
            default=option_labels_for_values(
                JOB_AREA_OPTIONS,
                stored_preferences["preferred_job_areas"]
            ),
            key="preferred_job_areas"
        )

        date_column, graduation_column = st.columns(2)

        with date_column:
            earliest_start_date = st.date_input(
                "Earliest available start date",
                value=metadata_date_value(
                    stored_candidate_metadata[
                        "availability"
                    ]["earliest_start_date"]
                ),
                key="earliest_start_date"
            )

        with graduation_column:
            graduation_date = st.date_input(
                "Graduation date",
                value=metadata_date_value(
                    stored_candidate_metadata[
                        "education"
                    ]["graduation_date"]
                ),
                key="graduation_date"
            )

        stored_travel = stored_candidate_metadata["travel"]
        travel_answer = st.selectbox(
            "Willing to travel?",
            optional_boolean_labels,
            index=optional_boolean_index(
                stored_travel["willing"]
            ),
            key="willing_to_travel"
        )
        maximum_travel = st.number_input(
            "Maximum travel percentage",
            min_value=0,
            max_value=100,
            value=stored_travel["maximum_percentage"],
            step=5,
            placeholder="Not provided",
            key="maximum_travel_percentage"
        )

        minimum_compensation = st.number_input(
            "Minimum compensation (annual USD, optional)",
            min_value=0,
            value=stored_candidate_metadata[
                "compensation"
            ]["minimum_amount"],
            step=1000,
            placeholder="Not provided",
            key="minimum_compensation"
        )

candidate_metadata = dict(stored_candidate_metadata)
work_authorization = dict(stored_work_authorization)
work_authorization.update(
    {
        "us_authorized": optional_boolean_value(
            authorization_answer
        ),
        "requires_sponsorship": optional_boolean_value(
            sponsorship_answer
        )
    }
)
candidate_metadata[
    "work_authorization"
] = work_authorization

location = dict(stored_location)
location.update(
    {
        "current": current_location,
        "willing_to_relocate": optional_boolean_value(
            relocation_answer
        )
    }
)
candidate_metadata["location"] = location

work_preferences = dict(stored_preferences)
work_preferences.update(
    {
        "arrangements": [
            WORK_ARRANGEMENT_OPTIONS[label]
            for label in arrangement_labels
        ],
        "job_types": [
            JOB_TYPE_OPTIONS[label]
            for label in job_type_labels
        ],
        "preferred_job_areas": [
            JOB_AREA_OPTIONS[label]
            for label in job_area_labels
        ]
    }
)
candidate_metadata["work_preferences"] = work_preferences

availability = dict(
    stored_candidate_metadata["availability"]
)
availability["earliest_start_date"] = normalize_optional_date(
    earliest_start_date
)
candidate_metadata["availability"] = availability

education = dict(stored_candidate_metadata["education"])
education["graduation_date"] = normalize_optional_date(
    graduation_date
)
candidate_metadata["education"] = education

travel = dict(stored_travel)
travel.update(
    {
        "willing": optional_boolean_value(travel_answer),
        "maximum_percentage": maximum_travel
    }
)
candidate_metadata["travel"] = travel

candidate_metadata["security_clearance"] = {
    **stored_candidate_metadata["security_clearance"],
    "status": CLEARANCE_OPTIONS[clearance_answer]
}
candidate_metadata["compensation"] = {
    **stored_candidate_metadata["compensation"],
    "minimum_amount": minimum_compensation
}
candidate_metadata = normalize_candidate_metadata(
    candidate_metadata
)

if candidate_metadata != stored_candidate_metadata:
    try:
        save_candidate_metadata(candidate_metadata)
    except OSError as error:
        st.warning(
            "Candidate profile details could not be saved locally: "
            f"{error}"
        )

st.caption(
    "Settings save automatically. Not provided values remain unknown "
    "and do not affect the qualification score."
)


# -------------------------
# Job Description
# -------------------------

job_description = st.text_area(
    "Job Description",
    height=400,
    placeholder=(
        "Paste the full job "
        "description here..."
    )
)


# -------------------------
# Analyze
# -------------------------

if st.button(
    "Analyze Job",
    type="primary",
    use_container_width=True
):

    if not job_description.strip():

        st.warning(
            "Please paste a job description."
        )

    elif (
        resume_source == "Upload a resume"
        and uploaded_resume is None
    ):

        st.warning(
            "Please upload a resume."
        )

    elif (
        resume_source == "Use last uploaded"
        and last_resume_metadata is None
    ):

        st.warning(
            "Please upload a resume first."
        )

    else:

        try:

            cleaned_job = clean_text(
                job_description
            )
            save_candidate_metadata(
                candidate_metadata
            )

            if resume_source == "Upload a resume":
                resume_data = uploaded_resume.getvalue()
                resume_name = uploaded_resume.name
                resume_text = extract_resume_text(
                    resume_data,
                    resume_name
                )
                save_last_resume(
                    resume_data,
                    resume_name
                )
            else:
                resume_data, resume_name = load_last_resume()
                resume_text = extract_resume_text(
                    resume_data,
                    resume_name
                )

            with st.spinner(
                "Analyzing your match..."
            ):

                analysis = cached_analysis(
                    cleaned_job,
                    resume_text,
                    candidate_metadata,
                    ANALYSIS_CACHE_VERSION
                )

            analysis["resume_name"] = resume_name

            st.session_state[
                "analysis"
            ] = analysis

            st.success(
                "Analysis complete!"
            )

        except Exception as error:

            st.error(
                f"Something went wrong: {error}"
            )


# -------------------------
# Results
# -------------------------

if "analysis" in st.session_state:

    analysis = st.session_state[
        "analysis"
    ]

    st.divider()

    st.subheader(
        analysis.get(
            "job_title",
            "Unknown Job"
        )
    )

    st.caption(
        analysis.get(
            "company",
            "Unknown Company"
        )
    )

    if analysis.get("resume_name"):
        st.caption(
            "Resume: "
            + analysis["resume_name"]
        )

    st.header("Qualification Match")

    score = analysis.get(
        "match_score",
        0
    )

    st.metric(
        "Qualification Match",
        f"{score}%"
    )

    st.progress(
        score / 100
    )

    st.caption(
        "Score is calculated automatically "
        "from resume-supported experience, skills, and job "
        "requirements. Eligibility and preferences are separate."
    )

    automatic_downgrades = analysis.get(
        "automatic_downgrades",
        []
    )

    if automatic_downgrades:
        st.info(
            "Deterministic validation corrected "
            f"{len(automatic_downgrades)} model match(es)."
        )

        with st.expander(
            "Show automatic match corrections"
        ):
            for downgrade in automatic_downgrades:
                st.markdown(
                    f"**{downgrade['requirement']}**"
                )
                st.write(
                    f"{status_name(downgrade['from'])} → "
                    f"{status_name(downgrade['to'])}"
                )
                st.caption(downgrade["reason"])

    # ---------------------
    # Required
    # ---------------------

    st.header(
        "Required Qualifications"
    )

    for item in analysis.get(
        "required",
        []
    ):

        icon = status_icon(
            item["status"]
        )

        st.markdown(
            f"### {icon} "
            f"{item['requirement']}"
        )

        st.write(
            "**Status:** "
            + status_name(
                item["status"]
            )
        )

        st.write(
            "**Evidence:** "
            + item["evidence"]
        )

    # ---------------------
    # Preferred
    # ---------------------

    preferred = analysis.get(
        "preferred",
        []
    )

    if preferred:

        st.header(
            "Preferred Qualifications"
        )

        for item in preferred:

            icon = status_icon(
                item["status"]
            )

            st.markdown(
                f"### {icon} "
                f"{item['requirement']}"
            )

            st.write(
                "**Status:** "
                + status_name(
                    item["status"]
                )
            )

            st.write(
                "**Evidence:** "
                + item["evidence"]
            )

    # ---------------------
    # Experience
    # ---------------------

    st.header(
        "Experience Fit"
    )

    st.write(
        analysis.get(
            "experience_fit",
            ""
        )
    )

    # ---------------------
    # Eligibility
    # ---------------------

    st.header("Eligibility")

    for item in analysis.get(
        "eligibility",
        {}
    ).get("items", []):
        st.markdown(
            f"{fit_status_icon(item['status'])} "
            f"**{item['label']}:** {item['detail']}"
        )

    st.caption(
        "Eligibility is evaluated from candidate-supplied settings "
        "and does not affect the qualification score."
    )

    # ---------------------
    # Preferences
    # ---------------------

    st.header("Preferences")

    for item in analysis.get(
        "preferences",
        {}
    ).get("items", []):
        st.markdown(
            f"{fit_status_icon(item['status'])} "
            f"**{item['label']}:** {item['detail']}"
        )

    st.caption(
        "Preference fit and mismatches are informational and do not "
        "affect the qualification score."
    )

    # ---------------------
    # Resume Improvements
    # ---------------------

    st.header(
        "Resume Improvements"
    )

    for improvement in analysis.get(
        "resume_improvements",
        []
    ):

        st.markdown(
            f"- {improvement}"
        )

    # ---------------------
    # Interview Prep
    # ---------------------

    st.header(
        "Interview Preparation"
    )

    for topic in analysis.get(
        "interview_topics",
        []
    ):

        st.markdown(
            f"- {topic}"
        )
