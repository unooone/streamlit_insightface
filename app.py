import streamlit as st
from streamlit_webrtc import (
    VideoProcessorBase,
    WebRtcMode,
    WebRtcStreamerContext,
    create_mix_track,
    create_process_track,
    webrtc_streamer,
)
import cv2, av, os, shutil, glob, threading, datetime
import numpy as np
from insightface.app import FaceAnalysis

#
PATH_MEDIA_DIR = "./media"
PATH_USERDATA_DIR = os.path.join(PATH_MEDIA_DIR, "userdata/")

ctx_streamer:WebRtcStreamerContext = None
lock_latest_img:threading.Lock = threading.Lock()
latest_img:cv2.Mat = None

list_user:list[str] = []
current_user:str = None
thre_similarity:float = 0.28

list_user_image:list=[]
list_user_feature:list = []

list_user = os.listdir(PATH_USERDATA_DIR)

# 類似度の算出
def compute_sim(feat1, feat2):
    return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))

# Events
def btn_add_user_clicked(username:str):
    global PATH_USERDATA_DIR
    if(username):
        path = os.path.join(PATH_USERDATA_DIR, username)
        if(not os.path.exists(path)):
            os.mkdir(os.path.join(PATH_USERDATA_DIR, username))
        st.session_state['current_user'] = username
def btn_del_user_clicked():
    global PATH_USERDATA_DIR
    username = st.session_state['current_user']
    path = os.path.join(PATH_USERDATA_DIR, username)
    if(os.path.exists(path)):
        shutil.rmtree(path)
    del st.session_state['current_user']
def btn_add_face_clicked():
    global ctx_streamer, current_user, PATH_USERDATA_DIR
    if(not current_user): 
        return  
    if(not ctx_streamer):
       return
    with lock_latest_img:
        if(latest_img is not None):
            now = datetime.datetime.now()
            path = os.path.join(PATH_USERDATA_DIR, current_user)
            path = os.path.join(path, f"{now.strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(path, latest_img)
def btn_del_face_clicked(filepath):
    if(os.path.exists(filepath)):
        os.remove(filepath)
def on_thre_similarity_change_value():
    st.write("しきい値を変更しました:", st.session_state.thre_similarity)
def on_user_feature_updated():
    st.write("ユーザー情報が読み込まれました:", st.session_state.current_user)

#
st.title("Streamlit-InsightFace DEMO")
st.write("FaceRecognition Test")

# Sidebar
selected_model_type = st.sidebar.selectbox("選択モデル", ['buffalo_s', 'buffalo_l'], key="model_type")
new_username = st.sidebar.text_input("新規ユーザー名")
st.sidebar.button("ユーザー追加", on_click=btn_add_user_clicked, args=[new_username])
st.sidebar.button("現在のユーザーを削除", on_click=btn_del_user_clicked)

# Main
if 'current_user' in st.session_state: 
    current_user = st.session_state.current_user
if 'thre_similarity' in st.session_state:
    thre_similarity = st.session_state.thre_similarity

with st.expander("設定", expanded=True):
    if current_user in list_user:
        st.selectbox("認証ユーザー選択", list_user, index=list_user.index(current_user), key="current_user")
    else:
        st.selectbox("認証ユーザー選択", list_user, disabled=len(list_user) < 1, key="current_user")
    #
    st.slider("本人判定値", min_value=0.0, max_value=1.0, step=0.01, value=thre_similarity, key="thre_similarity", on_change=on_thre_similarity_change_value)
    st.button("現在の画像を登録", on_click=btn_add_face_clicked, key="btn_add_photo")

# Initialize InsightFace
app = FaceAnalysis(name=selected_model_type, root='./')
app.prepare(ctx_id=0, det_size=(640, 480))

# Load user features
if(current_user):
    path_userdata = os.path.join(PATH_USERDATA_DIR, current_user)
    for item in glob.glob(f"{path_userdata}/*.*"):
        cv_img = cv2.imread(item)
        faces = app.get(cv_img)
        if(len(faces) > 0):
            list_user_image.append((item, cv_img))
            list_user_feature.append(faces[0].embedding)


# WebRTC Callback
def run_recognition(img, base_faces_emb:list, faces, thre=0.28):
    dimg = img.copy()
    for i in range(len(faces)):
        face = faces[i]
        box = face.bbox.astype(np.int64)
        sim_max = 0
        for base_emb_item in base_faces_emb:
            sim_max = max([sim_max, compute_sim(base_emb_item, face.embedding)])
        if(sim_max >= thre):
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)
        cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
        if face.kps is not None:
            kps = face.kps.astype(np.int64)
            #print(landmark.shape)
            for l in range(kps.shape[0]):
                color = (0, 0, 255)
                if l == 0 or l == 3:
                    color = (0, 255, 0)
                cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color,
                            2)
        if face.gender is not None and face.age is not None:
            cv2.putText(dimg,'%s, %d, %f'%(face.sex,face.age,sim_max), (box[0]-1, box[1]-4),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)
    return dimg
def video_frame_callback(frame):
    global list_user_feature, thre_similarity, lock_latest_img, latest_img
    cv_img = frame.to_ndarray(format="bgr24")
    with lock_latest_img:
        latest_img = cv_img.copy()
    faces = app.get(cv_img)
    cv_rimg = run_recognition(cv_img, list_user_feature, faces, thre_similarity)
    return av.VideoFrame.from_ndarray(cv_rimg, format="bgr24")

ctx_streamer = webrtc_streamer(key="camera", video_frame_callback=video_frame_callback, media_stream_constraints={"video": True, "audio": False},
    async_processing=True)

#
with st.expander("登録画像一覧", expanded=False):
    for idx, item in enumerate(list_user_image):
        filename, cv_img = item
        st.image(cv_img, caption=filename, width=240, channels='BGR')
        st.button("削除", on_click=btn_del_face_clicked, args=[filename], key=f"btn_del_{idx}")