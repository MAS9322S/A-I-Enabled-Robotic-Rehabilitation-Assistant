from datetime import datetime, timezone

import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from utils.database import Database
from utils.pose_processor import RehabilitationProcessor

st.set_page_config(
    page_title="AI Rehabilitation Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_rtc_configuration():
    """ICE configuration for browser -> cloud WebRTC connections."""
    ice_servers = [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
    try:
        turn_urls = st.secrets.get("TURN_URLS", "")
        turn_username = st.secrets.get("TURN_USERNAME", "")
        turn_credential = st.secrets.get("TURN_CREDENTIAL", "")
        if turn_urls and turn_username and turn_credential:
            urls = turn_urls if isinstance(turn_urls, list) else [
                item.strip() for item in str(turn_urls).split(",") if item.strip()
            ]
            ice_servers.append({
                "urls": urls,
                "username": turn_username,
                "credential": turn_credential,
            })
    except Exception:
        pass
    return {"iceServers": ice_servers}


@st.cache_resource
def get_database():
    return Database()


def reset_processor():
    exercise = st.session_state.get(
        "selected_exercise",
        "Elbow Flexion",
    )
    st.session_state.processor = RehabilitationProcessor()


def main():
    db = get_database()

    if "processor" not in st.session_state:
        reset_processor()

    st.title("🤖 AI-Enabled Robotic Rehabilitation Assistant")
    st.caption("Software-only rehabilitation prototype • Webcam + AI pose analysis")

    st.info(
        "Educational prototype only. It is not a medical device and its form score/feedback "
        "must not be used as a clinical diagnosis or treatment decision."
    )

    with st.sidebar:
        st.header("👤 Patient")
        patients = db.list_patients()
        patient_options = {f"{p['name']} (Age {p['age']})": p["id"] for p in patients}
        selected_label = st.selectbox(
            "Select patient",
            ["-- Select --"] + list(patient_options.keys()),
        )

        selected_patient_id = None if selected_label == "-- Select --" else patient_options[selected_label]

        with st.expander("➕ Create patient"):
            name = st.text_input("Name", key="new_patient_name")
            age = st.number_input("Age", min_value=1, max_value=120, value=21, step=1)
            if st.button("Create patient", use_container_width=True):
                if not name.strip():
                    st.error("Please enter a patient name.")
                else:
                    new_id = db.create_patient(name.strip(), int(age))
                    st.session_state.selected_patient_id = new_id
                    st.success("Patient created. Reloading list...")
                    st.rerun()

        st.divider()
        st.header("🏃 Exercise")
        exercise = st.selectbox( "Exercise",[
                    "Elbow Flexion",
                    "Shoulder Flexion",
                    "Shoulder Abduction",
                    "Knee Flexion",
                    ],)
        st.session_state.selected_exercise = exercise
            

        st.divider()
        st.caption("Deployment mode: " + db.mode_label)

    current_key = (selected_patient_id, exercise)
    if st.session_state.get("processor_key") != current_key:
        reset_processor()
        st.session_state.processor_key = current_key

    if not selected_patient_id:
        st.warning("Select or create a patient from the left sidebar to begin.")
        st.markdown(
            "### How it works\n"
            "1. Select a patient.\n"
            "2. Start the webcam.\n"
            "3. Keep your left shoulder, elbow and wrist visible.\n"
            "4. Slowly bend and straighten the elbow.\n"
            "5. Review repetitions, range and form feedback."
        )
        return

    patient = db.get_patient(selected_patient_id)
    if not patient:
        st.error("Patient record could not be found.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Patient", patient["name"])
    col2.metric("Age", patient["age"])
    col3.metric("Exercise", exercise)

    st.subheader("📹 Live rehabilitation session")
    st.write("Allow camera access when your browser asks. Stand far enough away that your upper body and left arm are visible.")

    processor = st.session_state.processor

    ctx = webrtc_streamer(
         key=f"rehab-{selected_patient_id}-{exercise}",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=get_rtc_configuration(),
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=lambda: processor,
        async_processing=True,
     )

    if ctx.state.playing:
        st.success("Camera connected. Perform slow, controlled elbow flexion and extension.")
    elif ctx.state.signalling:
        st.info("Connecting to the camera... Please allow camera access in the browser.")

    
    metrics = processor.get_metrics()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Elbow angle", f"{metrics['angle']:.1f}°")
    m2.metric("Repetitions", metrics["repetitions"])
    m3.metric("Form score", f"{metrics['form_score']:.0f}/100")
    m4.metric("Status", metrics["status"])

    if metrics["feedback"]:
        st.info(metrics["feedback"])

    if ctx.state.playing:
        st.caption("Keep the browser tab open while exercising. Stop the camera when the session is complete.")

    # Save once after the WebRTC stream has stopped and only when repetitions were recorded.
    if not ctx.state.playing and processor.has_session_data() and not processor.session_saved:
        db.save_session(
            patient_id=selected_patient_id,
            exercise=exercise,
            repetitions=processor.repetitions,
            avg_angle=processor.average_angle,
            max_range=processor.max_range,
            form_score=processor.form_score,
            feedback=processor.feedback,
        )
        processor.session_saved = True
        st.success("Session saved successfully.")

    st.divider()
    st.subheader("📈 Patient progress")
    sessions = db.list_sessions(selected_patient_id)
    if sessions:
        total_reps = sum(int(s["repetitions"]) for s in sessions)
        avg_score = sum(float(s["form_score"]) for s in sessions) / len(sessions)
        p1, p2, p3 = st.columns(3)
        p1.metric("Sessions", len(sessions))
        p2.metric("Total repetitions", total_reps)
        p3.metric("Average form score", f"{avg_score:.1f}/100")
        st.dataframe(sessions, use_container_width=True, hide_index=True)
    else:
        st.caption("No completed sessions yet.")

    with st.expander("ℹ️ About this prototype"):
        st.write(
            "The pose engine uses MediaPipe landmarks to estimate the left elbow angle. "
            "A small Random Forest model uses synthetic training data to estimate a demonstration form score. "
            "This is intended for a college software project and is not clinically validated."
        )


if __name__ == "__main__":
    main()
