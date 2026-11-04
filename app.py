import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from config import WEIGHTS,METRICS,CLASSES,DETAILS
from model_utils import load_model,read_image,predict
st.set_page_config(page_title='LeafLens AI | Plant Intelligence',page_icon='🌿',layout='wide',initial_sidebar_state='collapsed')
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Manrope:wght@500;700;800&display=swap');
html,body,.stApp {font-family:'DM Sans',sans-serif;color:#18392f}
.stApp{background:#f6f9f6}.block-container{max-width:1200px;padding-top:1.4rem}
h1,h2,h3{font-family:Manrope,sans-serif;letter-spacing:-.04em;color:#164b35}
#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important}
[data-testid="stHeader"]{background:transparent}
.hero{padding:45px;border-radius:28px;background:linear-gradient(125deg,#0b4332,#208e60);color:white;box-shadow:0 18px 40px #0b433223}
.hero h1{color:white!important;font-size:clamp(32px,4vw,57px);line-height:1.14}
.hero p{color:#e4f8eb;font-size:17px;max-width:680px}
.hero small{letter-spacing:2px;font-weight:800;color:#c8fce0}
.tag{display:inline-block;border:1px solid #ffffff66;border-radius:40px;padding:7px 13px;margin:8px 6px 0 0;color:white;font-size:12px}
[data-testid="stMetric"]{background:white;padding:18px;border:1px solid #e1eae4;border-radius:16px}
[data-testid="stMetric"] *,[data-testid="stMetricValue"] *,[data-testid="stMetricLabel"] *{color:#174a36!important;-webkit-text-fill-color:#174a36!important;opacity:1!important}
[data-baseweb="tab-list"]{gap:9px}button[data-baseweb="tab"]{border-radius:10px;background:#e9f1eb;padding:12px;color:#235943}
button[aria-selected="true"][data-baseweb="tab"]{background:#136e4a!important;color:white!important}
.stButton>button[kind="primary"]{background:#136e4a;border:0;color:white;border-radius:12px}
.card{background:white;border:1px solid #e1eae4;border-radius:18px;padding:20px;margin:12px 0}
.note{font-size:13px;color:#648076}
</style>""",unsafe_allow_html=True)
st.markdown('<h3 style="margin:0 0 18px">✳ LeafLens <span style="font-size:12px;color:#779e89">PLANT INTELLIGENCE STUDIO</span></h3>',unsafe_allow_html=True)
st.markdown("""<div class="hero"><small>✦ EC-422 · DEEP LEARNING PROJECT</small><h1>Healthier crops start<br>with a clearer picture.</h1><p>Discover how computer vision can help identify common tomato leaf diseases. Upload a photo and explore what a trained neural network sees.</p><span class="tag">🌿 Tomato leaf specialist</span><span class="tag">🧠 MobileNetV3 deep learning</span><span class="tag">⚡ Interactive model insights</span></div>""",unsafe_allow_html=True)
st.write('')
@st.cache_resource
def get_model():
    return load_model()
model=None
error=None
try:
    if WEIGHTS.is_file(): model=get_model()
except Exception as exc: error=str(exc)
metrics=json.loads(METRICS.read_text()) if METRICS.exists() else None
a,b,c=st.columns(3)
a.metric('Supported categories','05')
b.metric('Model architecture','MobileNetV3')
c.metric('Classifier status','Ready' if model is not None else 'Unavailable')
tabs=st.tabs(['🔍 Diagnose','📊 Model insights','📚 Disease library','🎓 How it works'])
with tabs[0]:
    st.write('')
    left,right=st.columns([1.05,.95],gap='large')
    with left:
        st.subheader('Analyze a leaf')
        st.caption('Upload a clear photo of one tomato leaf with good lighting.')
        f=st.file_uploader('Select tomato leaf image',type=['jpg','jpeg','png','webp'])
        im=None
        if f:
            try:
                im=read_image(f);st.image(im,use_container_width=True)
            except Exception as exc:st.error('Invalid image: '+str(exc))
        else: st.markdown('<div class="card" style="text-align:center;padding:55px"><h1>🍃</h1><h3>Every leaf tells a story</h3>Upload an image to begin.</div>',unsafe_allow_html=True)
        go_button=st.button('✨ Analyze leaf',type='primary',use_container_width=True,disabled=(im is None or model is None))
        st.caption('Educational screening, not a definitive agricultural diagnosis.')
    with right:
        st.subheader('AI assessment')
        if error:st.error('Could not load checkpoint: '+error)
        if model is None:st.warning('Trained checkpoint unavailable. Check models/leaflens_tomato.pt.')
        if go_button and im is not None:
            with st.spinner('Inspecting visual features...'):
                st.session_state['results']=predict(model,im);st.session_state['image']=f.name
        if f and st.session_state.get('image')==f.name and st.session_state.get('results') and model is not None:
            scores=st.session_state['results'];name,prob=scores[0];symbol,signs,action=DETAILS[name]
            st.success(f'{symbol} Predicted category: {name} · {prob:.1%} model score')
            st.caption('Softmax scores are not calibrated certainty.')
            for label,value in scores:
                st.markdown(f'**{label}** · {value:.1%}')
                st.progress(float(value))
            with st.expander('🔎 What to inspect next',expanded=True):
                st.write('Typical signs:',signs);st.write('Suggested action:',action)
            report='LeafLens AI — Educational Screening\nImage: '+f.name+'\nTop prediction: '+name+'\n\n'+'\n'.join(f'{n}: {v:.4f}' for n,v in scores)+'\nNot a confirmed diagnosis.'
            st.download_button('⬇ Download assessment report',report,'leaflens_report.txt')
        elif model is not None:st.markdown('<div class="card" style="padding:60px 20px;text-align:center"><h1>🧪</h1><h3>Ready when you are</h3>Your model scores and suggested next inspection will appear here.</div>',unsafe_allow_html=True)
with tabs[1]:
    st.subheader('Actual model evaluation')
    if metrics:
        a,b,c=st.columns(3)
        a.metric('Test accuracy',f"{metrics['accuracy']:.2%}")
        b.metric('Macro F1',f"{metrics['macro_f1']:.3f}")
        c.metric('Held-out test images',metrics['test_samples'])
        h=pd.DataFrame(metrics['history'])
        fig=go.Figure()
        fig.add_scatter(x=h.epoch,y=h.train_accuracy,mode='lines+markers',name='Training accuracy',line_color='#126948')
        fig.add_scatter(x=h.epoch,y=h.val_accuracy,mode='lines+markers',name='Validation accuracy',line_color='#e0a447')
        fig.update_layout(title='Learning over eight epochs',template='plotly_white',yaxis_tickformat='.0%',yaxis_range=[0,1],xaxis_title='Epoch')
        st.plotly_chart(fig,use_container_width=True)
        fig=go.Figure(go.Heatmap(z=metrics['confusion_matrix'],x=CLASSES,y=CLASSES,colorscale='Greens',text=metrics['confusion_matrix'],texttemplate='%{text}'))
        fig.update_layout(title='Confusion matrix (rows actual, columns predicted)',height=470)
        st.plotly_chart(fig,use_container_width=True)
        st.caption(metrics.get('split_note','Evaluation on held-out images from the dataset.'))
        st.download_button('⬇ Download evaluation JSON',json.dumps(metrics,indent=2),'leaflens_metrics.json')
    else:st.info('Evaluation results are not available in this deployment.')
with tabs[2]:
    st.subheader('Disease reference library')
    cols=st.columns(2)
    for i,(name,(symbol,signs,action)) in enumerate(DETAILS.items()):
        with cols[i%2]:
            st.markdown(f'<div class="card"><h3>{symbol} {name}</h3><p>{signs}</p><p class="note"><strong>Next step:</strong> {action}</p></div>',unsafe_allow_html=True)
with tabs[3]:
    st.subheader('How the AI works')
    for heading,desc in [('01 · Dataset','500 labeled tomato-leaf images representing 5 classes.'),('02 · Transfer learning','MobileNetV3 starts with general-purpose visual features, then learns tomato leaf categories.'),('03 · Evaluation','Held-out images assess generalization, using accuracy, macro F1, and a confusion matrix.'),('04 · Prediction','A new image is resized, normalized, classified, and shown with five scores.')]:
        with st.expander(heading,expanded=True):st.write(desc)
    st.warning('Limitations: Only five categories, small controlled-image dataset, and possible related images across splits. Real-world field accuracy may differ.')
    st.markdown('Dataset: [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) · CNN: MobileNetV3-Small · EC-422')
st.caption('LeafLens AI · Student deep-learning demonstration · EC-422')
