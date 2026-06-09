from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io, time

app = FastAPI(title="Waste Classifier")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

print("Loading model...")

# Gunakan SavedModel format, bukan .h5
# Format ini jauh lebih kompatibel lintas versi Keras/TensorFlow
MODEL = tf.saved_model.load('waste_model_savedmodel')
INFER = MODEL.signatures['serving_default']

IMG_SIZE = (224, 224)
print("Model siap!")

CLASS_MAP = {'0': 'Organic', '1': 'Inorganic'}
TIPS = {
    'Organic': {
        'color': '#27ae60',
        'desc' : 'Sampah organik berasal dari makhluk hidup dan dapat terurai alami.',
        'tips' : ['Buat kompos dari sisa makanan',
                  'Buang ke tempat sampah hijau (organik)',
                  'Bisa dijadikan pupuk tanaman']
    },
    'Inorganic': {
        'color': '#2980b9',
        'desc' : 'Sampah anorganik sulit terurai dan perlu didaur ulang.',
        'tips' : ['Pisahkan plastik, kertas, kaca',
                  'Bawa ke bank sampah terdekat',
                  'Kurangi penggunaan plastik sekali pakai']
    }
}

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert('RGB').resize(IMG_SIZE)
        arr = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)

        t0 = time.time()

        # Inferensi pakai SavedModel signature
        input_tensor = tf.constant(arr)
        output = INFER(input_tensor)

        # Ambil output — cek key yang tersedia
        output_key = list(output.keys())[0]
        pred = output[output_key].numpy()
        ms   = round((time.time() - t0) * 1000, 1)

        score      = float(pred[0][0])
        is_organic = score >= 0.5
        label = 'Organic' if is_organic else 'Inorganic'
        confidence = round(score * 100 if is_organic else (1 - score) * 100, 1)
        info       = TIPS[label]

        return JSONResponse({
            'success'   : True,
            'label'     : label,
            'confidence': confidence,
            'color'     : info['color'],
            'desc'      : info['desc'],
            'tips'      : info['tips'],
            'infer_ms'  : ms
        })
    except Exception as e:
        return JSONResponse(status_code=500,
                            content={'success': False, 'error': str(e)})

@app.get("/health")
async def health():
    return {"status": "ok", "model": "MobileNetV2 SavedModel"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)